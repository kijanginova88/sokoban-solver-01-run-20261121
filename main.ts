// Sokoban solver for make.fun's "Build the best Sokoban solver, 21 November 2026".
//
// The levels are a hallway with one to four rooms off it; a box pushed out of a
// room's door is lost, so every box stays in its room and a level's push-optimal
// count is the sum of its rooms' optima. This program therefore solves each room
// on its own, push-optimally, and walks the player between the rooms' doors.
// A room is solved by A* over pushes: a state is the sorted box squares plus the
// lowest square of the player's reachable region (flood fill around the boxes).
// The estimate is the Hungarian assignment of boxes to goals over BFS distances
// (walls only), which is admissible, so the first goal state A* pops is optimal.
// Dead states are pruned: boxes on dead squares (no goal reachable), boxes in a
// corner or a 2x2 block of walls, and boxes frozen on a wall run holding no goal.
// Budgets are counted in expansions, never the clock, so the two graded runs
// agree byte for byte. Node's built-in modules only; no network, no randomness.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets (work, never the clock).
const ROOM_ASTAR_BUDGET = 400_000;
const ROOM_GREEDY_BUDGET = 400_000;
const GLOBAL_BUDGET = 200_000;

// Directions in a fixed order, so every search and walk is deterministic.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];

type Level = {
  w: number;
  h: number;
  n: number;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(0, ...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, n, wall, goal, goals, boxes, player };
}

/** Per-level scratch: flood fills, walks and BFS distances. Single-threaded use. */
class Ctx {
  w: number;
  h: number;
  n: number;
  wall: Uint8Array;
  goal: Uint8Array;
  seen: Uint32Array;
  stamp = 0;
  q: Int32Array;
  parent: Int32Array;
  stepCh: Uint8Array;
  blocked: Uint8Array;
  regionLen = 0;
  regionMin = -1;

  constructor(level: Level) {
    this.w = level.w;
    this.h = level.h;
    this.n = level.n;
    this.wall = level.wall;
    this.goal = level.goal;
    this.seen = new Uint32Array(level.n);
    this.q = new Int32Array(level.n);
    this.parent = new Int32Array(level.n);
    this.stepCh = new Uint8Array(level.n);
    this.blocked = new Uint8Array(level.n);
  }

  isWall(i: number): boolean {
    return i < 0 || i >= this.n || this.wall[i] === 1;
  }

  /**
   * Flood fill from `start` over floor cells not marked in `blocked`. Returns a
   * stamp; the region's cells sit in q[0..regionLen), its lowest cell in regionMin.
   * Membership is tested with inRegion(cell, stamp) until the next flood or walk.
   */
  flood(start: number, blocked: Uint8Array): number {
    const stamp = ++this.stamp;
    const { w, n, wall, seen, q } = this;
    let head = 0;
    let tail = 0;
    q[tail++] = start;
    seen[start] = stamp;
    let min = start;
    while (head < tail) {
      const c = q[head++];
      if (c < min) min = c;
      for (let d = 0; d < 4; d++) {
        const nb = c + DY[d] * w + DX[d];
        if (nb < 0 || nb >= n || wall[nb] === 1 || blocked[nb] === 1 || seen[nb] === stamp) continue;
        seen[nb] = stamp;
        q[tail++] = nb;
      }
    }
    this.regionLen = tail;
    this.regionMin = min;
    return stamp;
  }

  inRegion(cell: number, stamp: number): boolean {
    return cell >= 0 && cell < this.n && this.seen[cell] === stamp;
  }

  /** BFS distances from `from` over cells where allowed[i] is 1; -1 is unreachable. */
  bfsDist(from: number, allowed: Uint8Array): Int16Array {
    const dist = new Int16Array(this.n).fill(-1);
    const { w, n, q, seen } = this;
    const stamp = ++this.stamp;
    let head = 0;
    let tail = 0;
    q[tail++] = from;
    seen[from] = stamp;
    dist[from] = 0;
    while (head < tail) {
      const c = q[head++];
      for (let d = 0; d < 4; d++) {
        const nb = c + DY[d] * w + DX[d];
        if (nb < 0 || nb >= n || allowed[nb] !== 1 || seen[nb] === stamp) continue;
        seen[nb] = stamp;
        dist[nb] = dist[c] + 1;
        q[tail++] = nb;
      }
    }
    return dist;
  }

  /** Shortest walk from `from` to `to` avoiding walls and blocked cells, or null. */
  walk(from: number, to: number, blocked: Uint8Array): string | null {
    if (from === to) return '';
    const { w, n, wall, seen, q, parent, stepCh } = this;
    const stamp = ++this.stamp;
    let head = 0;
    let tail = 0;
    q[tail++] = from;
    seen[from] = stamp;
    parent[from] = -1;
    while (head < tail) {
      const c = q[head++];
      for (let d = 0; d < 4; d++) {
        const nb = c + DY[d] * w + DX[d];
        if (nb < 0 || nb >= n || wall[nb] === 1 || blocked[nb] === 1 || seen[nb] === stamp) continue;
        seen[nb] = stamp;
        parent[nb] = c;
        stepCh[nb] = CH[d].charCodeAt(0);
        if (nb === to) {
          let out = '';
          let cur = nb;
          while (cur !== from) {
            out = String.fromCharCode(stepCh[cur]) + out;
            cur = parent[cur];
          }
          return out;
        }
        q[tail++] = nb;
      }
    }
    return null;
  }
}

type Push = { box: number; dir: number };

type SearchOpts = {
  ctx: Ctx;
  roomSet: Uint8Array; // cells a box may occupy
  goals: number[];
  boxes0: number[];
  player0: number;
  staticBlocked: Uint8Array; // fixed walking obstacles (other rooms' boxes)
  exitCell: number; // -1: none; else the door cell, reachable at the goal
  maxExpansions: number;
  greedy: boolean; // true: order by the estimate alone, to find any solution fast
};

function allPermutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

/**
 * Push-space A* (or greedy best-first when greedy) over one room, or over the
 * whole level when roomSet is every floor cell. Returns the push list, or null.
 */
function searchPushes(o: SearchOpts): Push[] | null {
  const { ctx, roomSet, goals, boxes0, player0, staticBlocked, exitCell, maxExpansions, greedy } = o;
  const w = ctx.w;
  const n = ctx.n;
  const wall = ctx.wall;
  const nB = boxes0.length;
  const goalSet = new Uint8Array(n);
  for (const g of goals) goalSet[g] = 1;

  // BFS distances from each goal over the cells a box may occupy.
  const dist: Int16Array[] = goals.map((g) => ctx.bfsDist(g, roomSet));
  const useHungarian = goals.length === nB && nB <= 7;
  const perms = useHungarian ? allPermutations(nB) : [];

  // The estimate depends on the boxes alone, so it is worth memoising.
  const hCache = new Map<string, number>();
  const heuristic = (bs: number[]): number => {
    const k = String.fromCharCode(...bs);
    const hit = hCache.get(k);
    if (hit !== undefined) return hit;
    let best = Infinity;
    if (useHungarian) {
      outer: for (const p of perms) {
        let sum = 0;
        for (let i = 0; i < nB; i++) {
          const d = dist[p[i]!]![bs[i]!]!;
          if (d < 0) continue outer;
          sum += d;
          if (sum >= best) continue outer;
        }
        if (sum < best) best = sum;
      }
    } else {
      // Too many boxes for the assignment: sum each box's nearest goal instead.
      let sum = 0;
      for (let i = 0; i < nB; i++) {
        let m = Infinity;
        for (let g = 0; g < goals.length; g++) {
          const d = dist[g]![bs[i]!]!;
          if (d >= 0 && d < m) m = d;
        }
        if (m === Infinity) {
          sum = Infinity;
          break;
        }
        sum += m;
      }
      best = sum;
    }
    hCache.set(k, best);
    return best;
  };

  const isWall = (i: number): boolean => i < 0 || i >= n || wall[i] === 1;
  // A cell a box can never slide onto: a wall, or a cell outside the room.
  const slideBlocked = (i: number): boolean => i < 0 || i >= n || wall[i] === 1 || roomSet[i] !== 1;

  // A box against a wall in direction (wx, wy) is frozen when it can only slide
  // along that wall, the run it slides on is walled off at both ends, and the
  // run holds no goal: then it can never reach one.
  const wallRunDead = (c: number, wx: number, wy: number): boolean => {
    const ow = wy * w + wx;
    const opA = wx * w - wy;
    const opB = -wx * w + wy;
    let a = c;
    while (!slideBlocked(a + opA) && isWall(a + opA + ow)) a += opA;
    let b = c;
    while (!slideBlocked(b + opB) && isWall(b + opB + ow)) b += opB;
    if (!slideBlocked(a + opA) || !slideBlocked(b + opB)) return false;
    for (let cell = b; ; cell += opA) {
      if (goalSet[cell] === 1) return false;
      if (cell === a) break;
    }
    return true;
  };

  const boxDead = (c: number): boolean => {
    if (goalSet[c] === 1) return false;
    const up = c - w;
    const down = c + w;
    const left = c - 1;
    const right = c + 1;
    const wU = isWall(up);
    const wD = isWall(down);
    const wL = isWall(left);
    const wR = isWall(right);
    if ((wU && wL) || (wU && wR) || (wD && wL) || (wD && wR)) return true;
    if (wR && wD && isWall(down + 1)) return true;
    if (wL && wD && isWall(down - 1)) return true;
    if (wR && wU && isWall(up + 1)) return true;
    if (wL && wU && isWall(up - 1)) return true;
    if (wU && wallRunDead(c, 0, -1)) return true;
    if (wD && wallRunDead(c, 0, 1)) return true;
    if (wL && wallRunDead(c, -1, 0)) return true;
    if (wR && wallRunDead(c, 1, 0)) return true;
    return false;
  };

  // Binary heap on [f, g, state], ties to the deeper state, then the earlier one.
  type Entry = [number, number, number];
  const less = (a: Entry, b: Entry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heap: Entry[] = [];
  const heapPush = (e: Entry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): Entry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  // A state is its key: the player's canonical square, then the sorted boxes.
  const stateKey = (playerCanon: number, bs: number[]): string =>
    String.fromCharCode(playerCanon, ...bs);
  const keys: string[] = [];
  const gScore: number[] = [];
  const parent: number[] = [];
  const pushBox: number[] = [];
  const pushDir: number[] = [];
  const best = new Map<string, number>();

  const setBlocked = (bs: number[]): void => {
    ctx.blocked.fill(0);
    for (let i = 0; i < n; i++) if (staticBlocked[i] === 1) ctx.blocked[i] = 1;
    for (const b of bs) ctx.blocked[b] = 1;
  };

  const startBoxes = boxes0.slice().sort((a, b) => a - b);
  const h0 = heuristic(startBoxes);
  if (h0 === Infinity) return null;
  setBlocked(startBoxes);
  ctx.flood(player0, ctx.blocked);
  const startCanon = ctx.regionMin;
  keys.push(stateKey(startCanon, startBoxes));
  gScore.push(0);
  parent.push(-1);
  pushBox.push(-1);
  pushDir.push(-1);
  best.set(keys[0]!, 0);
  heapPush([h0, 0, 0]);

  // The current state's walking obstacles, kept apart from ctx.blocked, which
  // the new state's flood reuses.
  const curBlocked = new Uint8Array(n);

  const reconstruct = (idx: number): Push[] => {
    const out: Push[] = [];
    for (let t = idx; t > 0; t = parent[t]!) out.push({ box: pushBox[t]!, dir: pushDir[t]! });
    out.reverse();
    return out;
  };

  let expansions = 0;
  while (heap.length > 0) {
    const [, g, idx] = heapPop();
    const key = keys[idx]!;
    if (best.get(key) !== idx) continue; // a better path to this state was found
    const boxes: number[] = [];
    for (let i = 1; i < key.length; i++) boxes.push(key.charCodeAt(i));
    const playerCanon = key.charCodeAt(0);
    let solved = true;
    for (const b of boxes) {
      if (goalSet[b] !== 1) {
        solved = false;
        break;
      }
    }
    if (solved) {
      if (exitCell < 0) return reconstruct(idx);
      setBlocked(boxes);
      const stamp = ctx.flood(playerCanon, ctx.blocked);
      if (ctx.inRegion(exitCell, stamp)) return reconstruct(idx);
      // Solved but the player cannot leave the room: keep looking.
    }
    if (expansions >= maxExpansions) break;
    expansions++;
    curBlocked.fill(0);
    for (let i = 0; i < n; i++) if (staticBlocked[i] === 1) curBlocked[i] = 1;
    for (const b of boxes) curBlocked[b] = 1;
    const stamp = ctx.flood(playerCanon, curBlocked);
    // Collect the pushes first: the new state's flood must not clobber the
    // region this state's pushes are tested against.
    const cands: { nb: number[]; b: number; d: number }[] = [];
    for (let i = 0; i < nB; i++) {
      const b = boxes[i]!;
      for (let d = 0; d < 4; d++) {
        const step = DY[d] * w + DX[d];
        const dest = b + step;
        if (roomSet[dest] !== 1 || wall[dest] === 1 || curBlocked[dest] === 1) continue;
        const stand = b - step;
        if (!ctx.inRegion(stand, stamp)) continue;
        const nb = boxes.slice();
        nb[i] = dest;
        nb.sort((x, y) => x - y);
        cands.push({ nb, b, d });
      }
    }
    for (const { nb, b, d } of cands) {
      let dead = false;
      for (const c of nb) {
        if (boxDead(c)) {
          dead = true;
          break;
        }
      }
      if (dead) continue;
      const h = heuristic(nb);
      if (h === Infinity) continue;
      // The player ends on the box's old square; the new state's canonical
      // square is the lowest square of the region around the new boxes.
      setBlocked(nb);
      ctx.flood(b, ctx.blocked);
      const newCanon = ctx.regionMin;
      const nkey = stateKey(newCanon, nb);
      const g2 = g + 1;
      const prev = best.get(nkey);
      if (prev !== undefined && gScore[prev]! <= g2) continue;
      const ni = keys.length;
      keys.push(nkey);
      gScore.push(g2);
      parent.push(idx);
      pushBox.push(b);
      pushDir.push(d);
      best.set(nkey, ni);
      heapPush([greedy ? h : g2 + h, g2, ni]);
    }
  }
  return null;
}

type Room = {
  door: number; // the room's door square
  entry: number; // the hallway square outside the door
  cells: number[]; // the room's floor squares, door included
  cellSet: Uint8Array;
  boxes: number[];
  goals: number[];
};

/**
 * Split a level into its rooms: the player starts on the hallway, each room
 * opens onto it through one door, and the room is the floor the door reaches
 * without crossing the hallway row. Null when the split does not cover the level.
 */
function detectRooms(level: Level, ctx: Ctx): Room[] | null {
  const { w, h, wall, boxes, goals, player } = level;
  const H = Math.floor(player / w);
  if (H < 1 || H >= h - 1) return null;
  const rooms: Room[] = [];
  for (const y of [H - 1, H + 1]) {
    if (y < 0 || y >= h) continue;
    for (let x = 0; x < w; x++) {
      const door = y * w + x;
      if (wall[door] === 1 || wall[H * w + x] === 1) continue;
      // The room: the floor the door reaches without crossing the hallway.
      ctx.blocked.fill(0);
      for (let i = 0; i < level.n; i++) if (Math.floor(i / w) === H) ctx.blocked[i] = 1;
      ctx.flood(door, ctx.blocked);
      const cells: number[] = [];
      for (let i = 0; i < ctx.regionLen; i++) cells.push(ctx.q[i]);
      cells.sort((a, b) => a - b);
      const cellSet = new Uint8Array(level.n);
      for (const c of cells) cellSet[c] = 1;
      const roomBoxes = boxes.filter((b) => cellSet[b] === 1);
      const roomGoals = goals.filter((g) => cellSet[g] === 1);
      if (roomBoxes.length === 0 || roomBoxes.length !== roomGoals.length) continue;
      if (cellSet[player] === 1) continue;
      // A room's door is unique: a component inside a room already found is no
      // new room, and a bigger component replaces the smaller one it covers.
      if (rooms.some((r) => cells.every((c) => r.cellSet[c] === 1))) continue;
      for (let i = rooms.length - 1; i >= 0; i--) {
        if (rooms[i]!.cells.every((c) => cellSet[c] === 1)) rooms.splice(i, 1);
      }
      rooms.push({ door, entry: H * w + x, cells, cellSet, boxes: roomBoxes, goals: roomGoals });
    }
  }
  // Every box and goal must sit in exactly one room.
  const covered = new Uint8Array(level.n);
  let boxCount = 0;
  for (const r of rooms) {
    for (const c of r.cells) covered[c] = 1;
    boxCount += r.boxes.length;
  }
  if (boxCount !== boxes.length) return null;
  for (const g of goals) if (covered[g] !== 1) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

/** Turn a push list into moves, walking the player between pushes. */
function pushesToMoves(
  ctx: Ctx,
  pushes: Push[],
  playerStart: number,
  boxes: number[],
): { moves: string; player: number } | null {
  const w = ctx.w;
  let moves = '';
  let player = playerStart;
  for (const p of pushes) {
    const step = DY[p.dir] * w + DX[p.dir];
    const stand = p.box - step;
    ctx.blocked.fill(0);
    for (const b of boxes) ctx.blocked[b] = 1;
    const walk = ctx.walk(player, stand, ctx.blocked);
    if (walk === null) return null;
    moves += walk + CH[p.dir];
    const at = boxes.indexOf(p.box);
    boxes[at] = p.box + step;
    player = p.box;
  }
  return { moves, player };
}

/** One A* over the whole level, used when the room split fails or a room does. */
function solveGlobal(level: Level, ctx: Ctx): string {
  const roomSet = new Uint8Array(level.n);
  for (let i = 0; i < level.n; i++) if (level.wall[i] === 0) roomSet[i] = 1;
  const pushes = searchPushes({
    ctx,
    roomSet,
    goals: level.goals,
    boxes0: level.boxes,
    player0: level.player,
    staticBlocked: new Uint8Array(level.n),
    exitCell: -1,
    maxExpansions: GLOBAL_BUDGET,
    greedy: false,
  });
  if (pushes === null) return '';
  const boxes = level.boxes.slice();
  const done = pushesToMoves(ctx, pushes, level.player, boxes);
  return done === null ? '' : done.moves;
}

/** Solve one level: room by room, or over the whole level when that fails. */
function solveLevel(level: Level): string {
  const ctx = new Ctx(level);
  if (level.boxes.length === 0) return '';
  const rooms = detectRooms(level, ctx);
  if (rooms !== null) {
    const boxes = level.boxes.slice();
    let player = level.player;
    let moves = '';
    for (const room of rooms) {
      const staticBlocked = new Uint8Array(level.n);
      for (const b of boxes) if (room.cellSet[b] !== 1) staticBlocked[b] = 1;
      ctx.blocked.fill(0);
      for (const b of boxes) ctx.blocked[b] = 1;
      const walkIn = ctx.walk(player, room.entry, ctx.blocked);
      if (walkIn === null) return solveGlobal(level, ctx);
      moves += walkIn;
      player = room.entry;
      let pushes = searchPushes({
        ctx,
        roomSet: room.cellSet,
        goals: room.goals,
        boxes0: room.boxes,
        player0: room.entry,
        staticBlocked,
        exitCell: room.door,
        maxExpansions: ROOM_ASTAR_BUDGET,
        greedy: false,
      });
      if (pushes === null) {
        pushes = searchPushes({
          ctx,
          roomSet: room.cellSet,
          goals: room.goals,
          boxes0: room.boxes,
          player0: room.entry,
          staticBlocked,
          exitCell: room.door,
          maxExpansions: ROOM_GREEDY_BUDGET,
          greedy: true,
        });
      }
      if (pushes === null) return solveGlobal(level, ctx);
      const done = pushesToMoves(ctx, pushes, player, boxes);
      if (done === null) return solveGlobal(level, ctx);
      moves += done.moves;
      player = done.player;
    }
    return moves;
  }
  return solveGlobal(level, ctx);
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const level = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
    if (level.player >= 0) moves = solveLevel(level);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
