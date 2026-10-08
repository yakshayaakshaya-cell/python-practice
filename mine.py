# adapters/mine.py
import time
import random
from adapter import Solver


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        t0 = time.time()
        BUDGET = 4.6
        deadline = t0 + BUDGET
        n = instance.size
        proc, rel, due = instance.proc, instance.release, instance.due
        wt, fam, setup = instance.weight, instance.fam, instance.setup
        rng = random.Random(2024)
        INF = float("inf")
        srow = [setup[f] for f in range(len(setup))]

        def evaluate(order):
            t = 0.0
            tard = 0.0
            prev = -1
            for b in order:
                if prev < 0:
                    s = rel[b]
                else:
                    s = t + srow[fam[prev]][fam[b]]
                    if rel[b] > s:
                        s = rel[b]
                t = s + proc[b]
                d = t - due[b]
                if d > 0:
                    tard += wt[b] * d
                prev = b
            return 4 * t + tard

        best = [INF, None]

        def consider(order, c):
            if c < best[0] - 1e-9:
                best[0] = c
                best[1] = list(order)
                try:
                    submit_candidate({"order": list(order)})
                except Exception:
                    pass

        ids = list(range(n))

        def greedy(alpha):
            rem = set(ids)
            order = []
            t = 0.0
            prev = -1
            while rem:
                bb, bk, bs = None, INF, 0.0
                for b in rem:
                    s = rel[b] if prev < 0 else max(t + srow[fam[prev]][fam[b]], rel[b])
                    f = s + proc[b]
                    k = (due[b] - f) / wt[b] + alpha * (s - (0 if prev < 0 else t))
                    if k < bk:
                        bk, bb, bs = k, b, s
                t = bs + proc[bb]
                prev = bb
                order.append(bb)
                rem.remove(bb)
            return order

        starts = [
            sorted(ids, key=lambda b: (due[b], rel[b])),
            sorted(ids, key=lambda b: (rel[b], due[b])),
            sorted(ids, key=lambda b: (due[b] - proc[b], rel[b])),
            sorted(ids, key=lambda b: (due[b] / wt[b], rel[b])),
        ]
        for a in (0.0, 0.5, 1.0, 2.0):
            starts.append(greedy(a))

        scored = []
        for o in starts:
            c = evaluate(o)
            consider(o, c)
            scored.append((c, o))
        scored.sort(key=lambda x: x[0])

        def insertion_ls(order, cur):
            order = list(order)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                idxs = ids[:]
                rng.shuffle(idxs)
                for i in idxs:
                    if time.time() >= deadline:
                        break
                    b = order[i]
                    rest = order[:i] + order[i + 1:]
                    m = n - 1
                    pt = [0.0] * (m + 1)
                    pd = [0.0] * (m + 1)
                    pp = [-1] * (m + 1)
                    t = 0.0
                    tard = 0.0
                    prev = -1
                    for j in range(m):
                        pt[j], pd[j], pp[j] = t, tard, prev
                        x = rest[j]
                        if prev < 0:
                            s = rel[x]
                        else:
                            s = t + srow[fam[prev]][fam[x]]
                            if rel[x] > s:
                                s = rel[x]
                        t = s + proc[x]
                        d = t - due[x]
                        if d > 0:
                            tard += wt[x] * d
                        prev = x
                    pt[m], pd[m], pp[m] = t, tard, prev

                    best_j, best_c = -1, cur
                    fb = fam[b]
                    for j in range(m + 1):
                        if j == i:
                            continue
                        t = pt[j]
                        tard = pd[j]
                        prev = pp[j]
                        if prev < 0:
                            s = rel[b]
                        else:
                            s = t + srow[fam[prev]][fb]
                            if rel[b] > s:
                                s = rel[b]
                        t = s + proc[b]
                        d = t - due[b]
                        if d > 0:
                            tard += wt[b] * d
                        prev = b
                        if tard + 4 * t >= best_c:
                            continue
                        ok = True
                        for k in range(j, m):
                            x = rest[k]
                            s = t + srow[fam[prev]][fam[x]]
                            if rel[x] > s:
                                s = rel[x]
                            t = s + proc[x]
                            d = t - due[x]
                            if d > 0:
                                tard += wt[x] * d
                            prev = x
                            if tard + 4 * t >= best_c:
                                ok = False
                                break
                        if ok:
                            c = tard + 4 * t
                            if c < best_c - 1e-9:
                                best_c, best_j = c, j
                    if best_j >= 0:
                        order = rest[:best_j] + [b] + rest[best_j:]
                        cur = evaluate(order)
                        improved = True
                        consider(order, cur)
            return order, cur

        def swap_ls(order, cur):
            order = list(order)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(n - 1):
                    if time.time() >= deadline:
                        break
                    for j in range(i + 1, n):
                        order[i], order[j] = order[j], order[i]
                        c = evaluate(order)
                        if c < cur - 1e-9:
                            cur = c
                            improved = True
                            consider(order, c)
                        else:
                            order[i], order[j] = order[j], order[i]
            return order, cur

        def full_ls(order, cur):
            while time.time() < deadline:
                order, c1 = insertion_ls(order, cur)
                order, c2 = swap_ls(order, c1)
                if c2 >= cur - 1e-9:
                    return order, c2
                cur = c2
            return order, cur

        def perturb(order):
            o = list(order)
            for _ in range(rng.randint(2, 4)):
                r = rng.random()
                if r < 0.6:
                    b = o.pop(rng.randrange(n))
                    o.insert(rng.randrange(n), b)
                elif r < 0.85:
                    i, j = rng.sample(range(n), 2)
                    o[i], o[j] = o[j], o[i]
                else:
                    L = rng.randint(2, min(5, n))
                    i = rng.randrange(n - L + 1)
                    blk = o[i:i + L]
                    del o[i:i + L]
                    p = rng.randrange(len(o) + 1)
                    o[p:p] = blk
            return o

        o, c = insertion_ls(scored[0][1], scored[0][0])
        o, c = full_ls(o, c)
        base, bcost = list(best[1]), best[0]
        stagn = 0
        si = 1
        while time.time() < deadline:
            cand = perturb(base)
            cc = evaluate(cand)
            cand, cc = insertion_ls(cand, cc)
            if cc < bcost - 1e-9:
                base, bcost, stagn = cand, cc, 0
                consider(cand, cc)
            else:
                stagn += 1
                if cc <= bcost * 1.005 and rng.random() < 0.15:
                    base, bcost = cand, cc
            if stagn > 40:
                if si < len(scored) and rng.random() < 0.5:
                    base, bcost = list(scored[si][1]), scored[si][0]
                    si += 1
                else:
                    base, bcost = list(best[1]), best[0]
                stagn = 0

        return {"order": best[1]}