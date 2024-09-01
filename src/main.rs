use proconio::input;
use std::collections::{HashMap, VecDeque, BinaryHeap, HashSet};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

struct Solver {
    n: usize,
    m: usize,
    t: usize,
    la: usize,
    lb: usize,
    g: HashMap<usize, Vec<usize>>,
    t_list: Vec<usize>,
    xy: Vec<(usize, usize)>,
    a: Vec<usize>,
    b: Vec<usize>,
    ans: Vec<String>,
    score: usize,
    lt: usize,
    rng: StdRng,
    match_rate: f64,
    switch_match_cnt: Vec<usize>,
    all_cnt: usize,
    a_opt: AOptimizer,
}

impl Solver {
    fn new(n: usize, m: usize, t: usize, la: usize, lb: usize, uv: Vec<(usize, usize)>, t_list: Vec<usize>, xy: Vec<(usize, usize)>) -> Solver {
        let mut g = HashMap::new();
        for v in 0..n {
            g.insert(v, vec![]);
        }
        for (u, v) in uv.iter() {
            let e = g.get_mut(u).unwrap();
            e.push(*v);
            let e = g.get_mut(v).unwrap();
            e.push(*u);
        }
        let mut a = vec![0; la];
        for i in 0..la {
            if i < n {
                a[i] = i;
            }
        }
        let b = vec![usize::MAX; lb];
        let ans: Vec<String> = Vec::new();
        let score = 0;
        let lt = 0;

        let seed: [u8; 32] = [0; 32];
        let rng = StdRng::from_seed(seed);
        let match_rate: f64 = 0.0;
        let switch_match_cnt: Vec<usize> = Vec::new();
        let all_cnt = 0;
        let a_opt = AOptimizer::new(n, la, lb);

        Solver { n, m, t, la, lb, g, t_list, xy, a, b, ans, score, lt, rng, match_rate, switch_match_cnt, all_cnt, a_opt }
    }

    fn get_path(&mut self) -> Vec<Vec<usize>> {
        let mut pathes: Vec<Vec<usize>> = Vec::new();

        // 経路の最適化(bfs)
        let mut t_pathes: Vec<Vec<Vec<usize>>> = vec![];
        let mut from = 0;
        let t_list = self.t_list.clone();
        for to in t_list.iter() {
            let pathes = self.bfs(from, *to);
            t_pathes.push(pathes);
            from = *to;
        }

        // BFSですでに通った経路を使用
        let mut from = 0;
        let mut all_visited: HashMap<(usize, usize), (usize, Vec<usize>)> = HashMap::new();  // key: (from, to), value: (cnt, path)
        for (i, to) in t_list.iter().enumerate() {
            let path = self.bfs2(from, *to, &all_visited);
            for j1 in 0..(path.len()-1) {
                for j2 in (j1+1)..path.len() {
                    let p1 = path[j1];
                    let p2 = path[j2];
                    let mut p = path[j1..(j2+1)].to_vec();
                    let e = all_visited.entry((p1, p2)).or_insert((0, p[1..p.len()].to_vec()));
                    e.0 += 1;
                    p.reverse();
                    let e = all_visited.entry((p2, p1)).or_insert((0, p[1..p.len()].to_vec()));
                    e.0 += 1;
                }
            }
            t_pathes[i].push(path);
            from = *to;
        }

        // 最短経路を取得
        let mut path: Vec<usize> = Vec::new();
        for i in 0..self.t {
            path.extend(t_pathes[i][0].to_vec());
        }
        pathes.push(path.clone());

        // path_setを最適化
        let path_set: HashSet<usize> = path.clone().into_iter().collect();
        let mut opt_len = path_set.len();
        let mut adop_path: Vec<usize> = vec![0; self.t];
        let mut v_freq: Vec<usize> = vec![0; self.n];
        for &p in path.iter() {
            v_freq[p] += 1;
        }
        let mut eval = 0.0;
        for &f in v_freq.iter() {
            if f == 0 { continue; }
            eval += (f as f64).ln();
        }
        
        println!("# befor opt_len: {}, eval: {}", opt_len, eval);

        fn change_path(before:  &Vec<usize>, after: &Vec<usize>, opt_len: usize, eval: f64, v_freq: &Vec<usize>) -> (bool, usize, f64, Vec<usize>) {
            let mut tmp_len = opt_len;
            let mut tmp_v_freq = v_freq.clone();

            for &p in before.iter() {
                tmp_v_freq[p] -= 1;
                if tmp_v_freq[p] == 0 {
                    tmp_len -= 1;
                }
            }
            for &p in after.iter() {
                tmp_v_freq[p] += 1;
                if tmp_v_freq[p] == 1 {
                    tmp_len += 1;
                }
            }
            let mut tmp_eval = 0.0;
            for &f in tmp_v_freq.iter() {
                if f == 0 { continue; }
                tmp_eval += (f as f64).ln();
            }
            let mut ret = false;
            if (opt_len > tmp_len) || (opt_len == tmp_len && eval > tmp_eval) {
            // if eval > tmp_eval {
                ret = true;
            }
            (ret, tmp_len, tmp_eval, tmp_v_freq)

        }

        for i in 0..self.t {
            for j in 1..t_pathes[0].len() {
                let before = &t_pathes[i][adop_path[i]];
                let after = &t_pathes[i][j];
                assert_eq!(before[before.len()-1], after[after.len()-1], "i: {}, j: {}, before: {:?}, after: {:?}", i, j, before, after);

                let ret = change_path(before, after, opt_len, eval, &v_freq);
                if ret.0 {
                    opt_len = ret.1;
                    eval = ret.2;
                    v_freq = ret.3;
                    adop_path[i] = j;
                }
            }
        }

        // 最適化後の経路を再取得
        let mut path: Vec<usize> = Vec::new();
        for i in 0..self.t {
            path.extend(t_pathes[i][adop_path[i]].to_vec());
        }
        pathes.push(path);
        println!("# after opt_len: {}, eval: {}", opt_len, eval);

        // 焼きなましで最適化
        let trial = 1000;
        let mut temperature = 0.0;
        for i in 0..trial {
            let i = self.rng.gen_range(0..self.t);
            let j = (adop_path[i]+1) % t_pathes[0].len();
            let before = &t_pathes[i][adop_path[i]];
            let after = &t_pathes[i][j];
            let ret = change_path(before, after, opt_len, eval, &v_freq);
            if ret.0 || self.rng.gen::<f64>() < temperature {
                opt_len = ret.1;
                eval = ret.2;
                v_freq = ret.3;
                adop_path[i] = j;
            }
            if trial % 100 == 0 {
                temperature /= 2.0;
            }
        }
        
        // 最適化後の経路を再取得
        let mut path: Vec<usize> = Vec::new();
        for i in 0..self.t {
            path.extend(t_pathes[i][adop_path[i]].to_vec());
        }
        pathes.push(path);
        println!("# after2 opt_len: {}, eval: {}", opt_len, eval);
        
        // 1の方の経路
        let mut path: Vec<usize> = Vec::new();
        for i in 0..self.t {
            path.extend(t_pathes[i][1].to_vec());
        }
        pathes.push(path);

        // bfs2の経路
        let mut path: Vec<usize> = Vec::new();
        for i in 0..self.t {
            path.extend(t_pathes[i][t_pathes[0].len()-1].to_vec());
        }
        pathes.push(path);

        pathes
    }

    fn optimize_a(&mut self, path: &Vec<usize>) {
        // 配列Aとpathで最適化
        self.a_opt.init(path);
        self.match_rate = self.a_opt.match_rate();
        self.a.clone_from(&self.a_opt.a);
        assert_eq!(self.a.len(), self.la, "Length of a is not equal LA. a: {}, LA: {}", self.a.len(), self.la);
    }

    fn bfs(&self, from: usize, to: usize) -> Vec<Vec<usize>> {
        let mut pathes: Vec<Vec<usize>> = vec![];

        // 1つ目追加
        let mut que: VecDeque<Vec<usize>> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        visited[from] = true;
        que.push_back(vec![from]);

        while let Some(path) = que.pop_front() {
            let &last = path.last().unwrap();
            if last == to {
                pathes.push(path[1..path.len()].to_vec());
                break;
            } else {
                for &neighbor in self.g.get(&last).unwrap().iter() {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        let mut new_path = path.clone();
                        new_path.push(neighbor);
                        que.push_back(new_path);
                    }
                }
            }
        }

        // 2つ目追加
        let mut que: VecDeque<Vec<usize>> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        visited[from] = true;
        que.push_back(vec![from]);

        while let Some(path) = que.pop_front() {
            let &last = path.last().unwrap();
            if last == to {
                pathes.push(path[1..path.len()].to_vec());
                break;
            } else {
                for &neighbor in self.g.get(&last).unwrap().iter().rev() {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        let mut new_path = path.clone();
                        new_path.push(neighbor);
                        que.push_back(new_path);
                    }
                }
            }
        }

        pathes
    }

    fn bfs2(&self, from: usize, to: usize, all_visited: &HashMap<(usize, usize), (usize, Vec<usize>)>) -> Vec<usize> {
        let mut que: VecDeque<Vec<usize>> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        let mut ret: Vec<usize> = Vec::new();
        visited[from] = true;
        que.push_back(vec![from]);

        while let Some(path) = que.pop_front() {
            let &last = path.last().unwrap();
            if all_visited.contains_key(&(last, to)) {
                let (cnt, p) = all_visited.get(&(last, to)).unwrap();
                let mut path = path;
                path.extend(p);
                return path[1..path.len()].to_vec();
            }
            if last == to {
                ret = path[1..path.len()].to_vec();
                break;
            } else {
                // すでに使った経路を選択(利用回数が最大の経路)
                let mut max_cnt: usize = 0;
                let mut opt_path: Vec<usize> = Vec::new();
                for &neighbor in self.g.get(&last).unwrap().iter() {
                    if all_visited.contains_key(&(neighbor, to)) {
                        let (cnt, p) = all_visited.get(&(neighbor, to)).unwrap();
                        if *cnt > max_cnt {
                            max_cnt = *cnt;
                            opt_path = vec![neighbor];
                            opt_path.extend(p);
                        }
                    }
                }
                if max_cnt > 0 {
                    let mut path = path;
                    path.extend(opt_path);
                    return path[1..path.len()].to_vec();
                }

                // 通常のDFS
                for &neighbor in self.g.get(&last).unwrap().iter() {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        let mut new_path = path.clone();
                        new_path.push(neighbor);
                        que.push_back(new_path);
                    }
                }
            }
        }

        ret
    }

    fn solve(&mut self) {
        let pathes = self.get_path();

        let mut opt_score = usize::MAX;
        let mut opt_path: Vec<usize> = Vec::new();
        for path in pathes.iter() {
            self.clear();
            // 最適な経路で実施
            self.optimize_a(path);
            self.lt = path.len();
            // println!("# path: {:?}", path);
            for (i, p) in path.iter().enumerate() {
                let target = path[i..path.len().min(i+self.lb)].to_vec();
                if !self.can_move(p, &self.b) {
                    let (l, s_a, s_b, match_cnt) = self.switch(&target);
                    self.switch_op(l, s_a, s_b);
                    self.switch_match_cnt.push(match_cnt);
                    self.all_cnt += self.lb;
                }
                self.r#move(p);
            }
            let match_rate = self.switch_match_cnt.iter().sum::<usize>() as f64 / self.all_cnt as f64;
            let score = self.score;
            if opt_score > score {
                opt_score = score;
                opt_path = path.clone();
            }
            println!("# match_rate: {}, score: {}", match_rate, score);
        }

        // 最適な経路で実施
        self.clear();
        let path = &opt_path;
        self.optimize_a(path);
        self.lt = path.len();
        println!("# path: {:?}", path);
        for (i, p) in path.iter().enumerate() {
            let target = path[i..path.len().min(i+self.lb)].to_vec();
            if !self.can_move(p, &self.b) {
                let (l, s_a, s_b, match_cnt) = self.switch(&target);
                self.switch_op(l, s_a, s_b);
                self.switch_match_cnt.push(match_cnt);
                self.all_cnt += self.lb;
            }
            self.r#move(p);
        }

    }

    fn clear(&mut self) {
        self.b = vec![usize::MAX; self.lb];
        self.ans = Vec::new();
        self.score = 0;
        self.lt = 0;

        self.match_rate = 0.0;
        self.switch_match_cnt = Vec::new();
        self.all_cnt = 0;
        self.a_opt = AOptimizer::new(self.n, self.la, self.lb);
    }

    fn switch(&mut self, target: &[usize]) -> (usize, usize, usize, usize) {
        let (min_i, max_i, match_cnt) = self.a_opt.r#match(target);
        let l = max_i - min_i + 1;
        let s_a = min_i;
        let s_b = 0;
        (l, s_a, s_b, match_cnt)
    }

    fn switch_op(&mut self, l: usize, s_a: usize, s_b: usize) {
        // println!("# s {} {} {}", l, s_a, s_b);
        self.b[s_b..(s_b+l)].clone_from_slice(&self.a[s_a..(s_a+l)]); 
        self.ans.push(format!("s {} {} {}", l, s_a, s_b));
        self.score += 1;
    }

    fn can_move(&self, p: &usize, b: &[usize]) -> bool {
        b.contains(p)
    }
    
    fn r#move(&mut self, p: &usize) {
        // println!("# m {}", p);
        self.ans.push(format!("m {}", p));
    }
    
    fn ans(self) {
        println!("{}", self.a.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        for a in self.ans.iter() {
            println!("{}", a);
        }
        let match_rate = self.switch_match_cnt.iter().sum::<usize>() as f64 / self.all_cnt as f64;
        let path_set: HashSet<usize> = self.a_opt.path.clone().into_iter().collect();
        eprintln!("{{ \"M\": {}, \"LA\": {}, \"LB\": {}, \"score\": {}, \"lt_lb\": {}, \"path_set_len\": {}, \"pred_match_rate\": {}, \"actual_match_rate\": {} }}", self.m, self.la, self.lb, self.score, (self.lt-1)/self.lb+1, path_set.len(), self.match_rate, match_rate);
    }
}

fn main() {
    input! {
        n: usize,
        m: usize,
        t: usize,
        la: usize,
        lb: usize,
        uv: [(usize, usize); m],
        t_list: [usize; t],
        xy: [(usize, usize); n],
    }

    let mut solver = Solver::new(n, m, t, la, lb, uv, t_list, xy);
    solver.solve();
    solver.ans();
}

struct AOptimizer {
    n: usize,
    la: usize,
    lb: usize,
    path: Vec<usize>,
    a: Vec<usize>,
    path_freq: Vec<HashMap<usize, usize>>,
    all_path_freq: Vec<usize>,
}

impl AOptimizer {
    fn new(n: usize, la: usize, lb: usize) -> AOptimizer {
        let path: Vec<usize> = Vec::new();
        let a: Vec<usize> = Vec::new();
        let path_freq: Vec<HashMap<usize, usize>> = Vec::new();
        let all_path_freq: Vec<usize> = Vec::new();
        AOptimizer { n, la, lb, path, a, path_freq, all_path_freq }
    }
    
    fn init(&mut self, path: &Vec<usize>) {
        self.path.clone_from(path);
        self.get_path_freq();  // pathにおける各都市の隣接都市情報を取得
        self.init_a();
    }

    fn init_a(&mut self) {
        // 配列Aとpathの一致率が一番よくなるように最適化する
        let path = &self.path;
        let path_set: HashSet<usize> = path.clone().into_iter().collect();  // 最初の都市がない経路から作成

        // v9の改善(都市の繋がりの件数を取得)
        let mut p_freq: Vec<Vec<HashMap<(usize, usize), usize>>> = vec![vec![HashMap::new(); self.lb]; self.n];  // [P][l][(pre, pl)] = cnt
        println!("# path len: {}, path_set len: {}", path.len(), path_set.len());
        // println!("# path: {:?}", path);
        for (i, p) in path[0..(path.len()-1)].iter().enumerate() {
            let mut pre = *p;
            for l in 0..self.lb.min(path.len()-i) {
                let pl = path[i+l];
                let e = p_freq[*p][l].entry((pre, pl)).or_insert(0);
                *e += 1;
                pre = pl;
            }
        }
        let p = path[path.len()-1];
        let _ = p_freq[p][0].entry((p, p)).or_insert(1);

        // 配列Aの1個目を追加(一番最初の都市に対して最適な配列を設置)
        let mut a: Vec<usize> = Vec::new();
        let mut heap: BinaryHeap<(usize, usize, usize, usize)> = BinaryHeap::new();  // (cnt, pre, pl, l)
        let mut added: Vec<bool> = vec![false; self.n];
        let p = path[0];
        heap.push((*p_freq[p][0].get(&(p, p)).unwrap(), p, p, 0));
        while a.len() < self.lb && !heap.is_empty() {
            let (_, _, pl, l) = heap.pop().unwrap();
            if !added[pl] {
                a.push(pl);
                added[pl] = true;
            }
            if l == self.lb-1 { continue; }
            for ((pre2, pl2), cnt) in p_freq[p][l+1].iter() {
                if pre2 != &pl && !added[*pl2] { continue; }
                heap.push((*cnt, pl, *pl2, l+1));
            }
        }

        // 以降は後ろLBの範囲が最大となる、まだ追加していない都市を追加していく
        while a.len() < path_set.len() {
            let mut opt_rate = f64::MIN; 
            let mut opt_p = usize::MAX;
            for p in 0..self.n {
                if !path_set.contains(&p) || added[p] { continue; }
                let mut ln_rate = 0.0;
                let si = if a.len()+1 > self.lb { a.len()+1-self.lb } else { 0 };
                let ei = a.len();
                let mut b = a[si..ei].to_vec();
                // b.push(p);
                let rate = self.calc_rate(p, &b, &p_freq);
                /*
                ln_rate += rate.ln();
                for i in 0..(self.lb-1) {
                    if self.lb + i > self.la { break; }
                    let pi = a[si+i];
                    b.pop();
                    b.push(pi);
                    let rate = self.calc_rate(pi, &b, &p_freq);
                    ln_rate += rate.ln();
                }
                */
                if opt_rate <= rate {
                    opt_rate = rate ;
                    opt_p = p;
                }
            }
            a.push(opt_p);
            added[opt_p] = true;
        }
        // println!("# a: {:?}", a);

        // 残りを埋める
        let mut max_freq: BinaryHeap<(usize, usize)> = BinaryHeap::new();
        for p in path_set.iter() {
            max_freq.push((*p_freq[*p][0].get(&(*p, *p)).unwrap(), *p));
        }
        while a.len() < self.la {
            let (_, p) = max_freq.pop().unwrap();
            let mut heap: BinaryHeap<(usize, usize, usize, usize)> = BinaryHeap::new();  // (cnt, pre, pl, l)
            let mut added: Vec<bool> = vec![false; self.n];
            heap.push((*p_freq[p][0].get(&(p, p)).unwrap(), p, p, 0));
            let mut cnt = 0;
            while cnt < self.lb && !heap.is_empty() {
                let (_, _, pl, l) = heap.pop().unwrap();
                if !added[pl] {
                    a.push(pl);
                    added[pl] = true;
                    cnt += 1;
                    if a.len() == self.la { break; }
                }
                if l == self.lb-1 { continue; }
                for ((pre2, pl2), cnt) in p_freq[p][l+1].iter() {
                    if pre2 != &pl && !added[*pl2] { continue; }
                    heap.push((*cnt, pl, *pl2, l+1));
                }
            }

        }

        for i in path_set.iter() {
            assert!(a.contains(i), "# not found: {}", i);
        }
        self.a = a;
    }

    fn calc_rate(&self, p: usize, b: &[usize], p_freq: &Vec<Vec<HashMap<(usize, usize), usize>>>) -> f64 {
        let mut cnt = 0;
        let all_cnt = p_freq[p][0].get(&(p, p)).unwrap() * self.lb;
        let mut target: HashMap<usize, (usize, usize)> = HashMap::new();  // 入っているといい都市(key: pl, value: (l, cnt)
        // 最初は1階層目を追加
        for ((_, pl), c) in p_freq[p][1].iter() {
            target.insert(*pl, (1, *c));
        }
        for i in 0..b.len() {
            let bi = b[i];
            if target.contains_key(&bi) {
                let (l, c) = *target.get(&bi).unwrap();
                cnt += c;
                if l == self.lb-1 { continue; }
                for ((pre, pl), c) in p_freq[p][l+1].iter() {
                    if pre != &bi { continue; }
                    let e = target.entry(*pl).or_insert((l+1, *c));
                    e.1 += c;
                }
            }
        }

        cnt as f64 / all_cnt as f64
    }

    fn get_path_freq(&mut self) {
        self.path_freq = vec![HashMap::new(); self.n];
        self.all_path_freq = vec![1; self.n];  // all_path_freqの初期値は1(evalの初期値が1のため)
        let mut path: Vec<usize> = vec![0];  // 最初の都市を追加
        path.extend(&self.path);
        for i in 0..(path.len()-1) {
            let t1 = path[i];
            let t2 = path[i+1];
            let e = self.path_freq[t1].entry(t2).or_insert(0);  // 隣接する都市の出現数
            *e += 1;
            // let e = self.path_freq[t2].entry(t1).or_insert(0);  // 隣接する都市の出現数
            // *e += 1;
        }

        for i in 0..self.n {
            for (_, cnt) in self.path_freq[i].iter() {
                self.all_path_freq[i] += cnt;
            }
        }
    }

    fn r#match(&self, target: &[usize]) -> (usize, usize, usize) {
        let mut min_i = usize::MAX;
        let mut max_i = 0;
        let mut match_cnt = 0;
        let mut candidates = vec![(min_i, max_i)];
        for t in target.iter() {
            let indices: Vec<usize> = self.a[0..self.la].iter().enumerate().filter(|(_, &x)| x == *t).map(|(i, _)| i).collect();
            let mut tmp: Vec<(usize, usize)> = Vec::new();
            for (min_i, max_i) in candidates.iter() {
                for index in indices.iter() {
                    let dist = max_i.max(index) - min_i.min(index) + 1;
                    if dist > self.lb {
                        continue;
                    } else {
                        tmp.push((*min_i.min(index), *max_i.max(index)));
                    }
                }
            }
            if tmp.is_empty() { break; } else {
                candidates = tmp;
                match_cnt += 1;
            }
        }
        (min_i, max_i) = candidates[0];
        (min_i, max_i, match_cnt)
    }

    fn match_rate(&self) -> f64 {
        let mut match_cnt = 0;
        let mut all_cnt = 0;
        for i in 0..self.path.len() {
            let target = self.path[i..(i+self.lb).min(self.path.len())].to_vec();
            let (_, _, cnt) = self.r#match(&target);
            match_cnt += cnt;
            all_cnt += target.len();
        }
        match_cnt as f64 / all_cnt as f64
    }

}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Solver {
        let n = 7;
        let m = 9;
        let t = 3;
        let la = 7;
        let lb = 4;
        let uv = vec![
            (0 ,1), (0 ,2), (0 ,3), (1 ,2), (2 ,3),
            (3 ,4), (4 ,5), (5 ,6), (6 ,0),
        ];
        let t_list = vec![4, 1, 5];
        let xy = vec![
            (100, 0), (200, 0), (200, 100), (100, 100),
            (0, 200), (0, 100), (0, 0),
        ];
        Solver::new(n, m, t, la, lb, uv, t_list, xy)
    }
    
    #[test]
    fn test_switch() {
        let mut solver = setup();
        assert_eq!(solver.a, [0, 1, 2, 3, 4, 5, 6]);
        solver.a_opt.a = vec![0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0];
        let (l, s_a, s_b, match_cnt) = solver.switch(&[3, 0, 2, 4]);
        solver.switch_op(l, s_a, s_b);
        assert_eq!(solver.b, [0, 1, 2, 3]);
        assert_eq!(match_cnt, 3);
    }

    #[test]
    fn test_bfs() {
        let solver = setup();
        let pathes = solver.bfs(1, 3);
        assert_eq!(pathes.len(), 2);
        assert_eq!(pathes[0], [0, 3]);
        assert_eq!(pathes[1], [2, 3]);
    }
}
