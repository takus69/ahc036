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

    fn get_path(&self) -> Vec<usize> {
        // 最適な経路を取得する
        let mut from = 0;
        let t_list = self.t_list.clone();
        let mut path: Vec<usize> = Vec::new();
        for to in t_list.iter() {
            let pathes = self.bfs(from, *to, 1);
            assert_ne!(pathes[0][0], from);
            assert_eq!(pathes[0][pathes[0].len()-1], *to);
            path.extend(pathes[0].to_vec());
            from = *to;
        }

        path
    }

    fn optimize_a(&mut self, path: &Vec<usize>) {
        // 配列Aとpathの一致率を算出
        self.a_opt.init(path);
        self.match_rate = self.a_opt.match_rate();
        println!("# match_rate: {}", self.match_rate);
        println!("# path: {:?}", path);

        // 配列Aを最適化
        // self.a_opt.optimize(1000);
        self.a.clone_from(&self.a_opt.a);
        assert_eq!(self.a.len(), self.la, "Length of a is not equal LA. a: {}, LA: {}", self.a.len(), self.la);

        // 最適化後の配列Aとpathの一致率を算出
        self.match_rate = self.a_opt.match_rate();
        println!("# match_rate: {}", self.match_rate);

        println!("# path: {}, lb: {}", path.len(), self.lb);
        println!("# a: {:?}", self.a);
    }

    fn bfs(&self, from: usize, to: usize, max_cnt: usize) -> Vec<Vec<usize>> {
        let mut pathes: Vec<Vec<usize>> = Vec::new();
        let mut prev: Vec<usize> = vec![usize::MAX; self.n];
        let mut que: VecDeque<usize> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        que.push_front(from);
        while !que.is_empty() {
            let u = que.pop_back().unwrap();
            if u == to {
                let path = reconstruct(from, to, &prev);
                pathes.push(path);
                if pathes.len() >= max_cnt {
                    break;
                } else {
                    visited[to] = false;
                    prev[to] = usize::MAX;
                    continue;
                }
            }
            for v in self.g.get(&u).unwrap().iter() {
                if visited[*v] { continue; }
                que.push_front(*v);
                visited[*v] = true;
                prev[*v] = u;
            }
        }

        // 経路復元
        fn reconstruct(from: usize, to: usize, prev: &Vec<usize>) -> Vec<usize> {
            let mut path: Vec<usize> = Vec::new();
            path.push(to);
            let mut next = to;
            while next != from {
                next = prev[next];
                path.push(next);
            }
            path.pop();  // fromを経路から削除
            path.reverse();

            path
        }

        pathes
    }

    fn solve(&mut self) {
        let path = self.get_path();
        self.optimize_a(&path);

        // 最適な経路で実施
        self.lt = path.len();
        println!("# path: {:?}", path);
        for (i, p) in path.iter().enumerate() {
            let target = path[i..path.len().min(i+self.lb)].to_vec();
            if !self.can_move(p, &self.b) {
                let (l, s_a, s_b, match_cnt) = self.switch(&target);
                self.switch_op(l, s_a, s_b);
                self.switch_match_cnt.push(match_cnt);
                self.all_cnt += self.lb;
                println!("# target: {:?}, b: {:?}", target, self.b);
            }
            self.r#move(p);
        }
    }

    fn switch(&mut self, target: &[usize]) -> (usize, usize, usize, usize) {
        let (min_i, max_i, match_cnt) = self.a_opt.r#match(target);
        let l = max_i - min_i + 1;
        let s_a = min_i;
        let s_b = 0;
        (l, s_a, s_b, match_cnt)
    }

    fn switch_op(&mut self, l: usize, s_a: usize, s_b: usize) {
        println!("# s {} {} {}", l, s_a, s_b);
        self.b[s_b..(s_b+l)].clone_from_slice(&self.a[s_a..(s_a+l)]); 
        self.ans.push(format!("s {} {} {}", l, s_a, s_b));
        self.score += 1;
    }

    fn can_move(&self, p: &usize, b: &[usize]) -> bool {
        b.contains(p)
    }
    
    fn r#move(&mut self, p: &usize) {
        println!("# m {}", p);
        self.ans.push(format!("m {}", p));
    }
    
    fn ans(self) {
        println!("{}", self.a.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        for a in self.ans.iter() {
            println!("{}", a);
        }
        let match_rate  =self.switch_match_cnt.iter().sum::<usize>() as f64 / self.all_cnt as f64;
        eprintln!("{{ \"M\": {}, \"LA\": {}, \"LB\": {}, \"score\": {}, \"lt_lb\": {}, \"pred_match_rate\": {}, \"actual_match_rate\": {} }}", self.m, self.la, self.lb, self.score, (self.lt-1)/self.lb+1, self.match_rate, match_rate);
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
    match_cnt: Vec<HashMap<usize, usize>>,
    eval_freq: Vec<usize>,
    eval: f64,
    rng: StdRng,
    must_a: HashMap<usize, usize>,
    g: HashMap<usize, Vec<usize>>,
}

impl AOptimizer {
    fn new(n: usize, la: usize, lb: usize) -> AOptimizer {
        let path: Vec<usize> = Vec::new();
        let a: Vec<usize> = Vec::new();
        let path_freq: Vec<HashMap<usize, usize>> = Vec::new();
        let all_path_freq: Vec<usize> = Vec::new();
        let match_cnt: Vec<HashMap<usize, usize>> = Vec::new();
        let eval_freq: Vec<usize> = Vec::new();
        let eval = 1.0;
        let seed: [u8; 32] = [0; 32];
        let rng = StdRng::from_seed(seed);
        let must_a: HashMap<usize, usize> = HashMap::new();
        let g: HashMap<usize, Vec<usize>> = HashMap::new();
        AOptimizer { n, la, lb, path, a, path_freq, all_path_freq, match_cnt, eval_freq, eval, rng, must_a, g }
    }
    
    fn init(&mut self, path: &Vec<usize>) {
        self.path.clone_from(path);
        self.get_path_freq();  // pathにおける各都市の隣接都市情報を取得
        self.init_a();


        // 配列aのLB範囲に存在する経路の隣接都市の個数
        self.match_cnt = vec![HashMap::new(); self.n];
        self.eval_freq = vec![1; self.n];
        for i in 0..self.la {
            let ai = self.a[i];
            for j in 1..self.lb {
                let aj = self.a[(i+j)%self.la];
                if ai == aj { continue; }
                if !self.path_freq[ai].contains_key(&aj) { continue; }
                let e = self.match_cnt[ai].entry(aj).or_insert(0);
                *e += 1;
                if *e == 1 {
                    self.eval_freq[ai] += self.path_freq[ai].get(&aj).unwrap();
                }
                if !self.path_freq[aj].contains_key(&ai) { continue; }
                let e = self.match_cnt[aj].entry(ai).or_insert(0);
                *e += 1;
                if *e == 1 {
                    self.eval_freq[aj] += self.path_freq[aj].get(&ai).unwrap();
                }
            }
        }

        // evalの初期計算
        let mut path_set: HashSet<usize> = path.clone().into_iter().collect();
        for ai in path_set.iter() {
            self.must_a.insert(*ai, 0);
        }
        for ai in self.a.iter() {
            if !path_set.contains(ai) { continue; }
            let e = self.must_a.get_mut(ai).unwrap();
            *e += 1;
        }
        path_set.insert(0);
        let mut eval: f64 = 1.0;
        for pi in path_set.iter() {
            eval *= self.eval_freq[*pi] as f64 / self.all_path_freq[*pi] as f64;
        }
        println!("# eval: {}", eval);
        self.eval = eval;


    }

    fn init_a(&mut self) {
        // 配列Aとpathの一致率が一番よくなるように最適化する
        let path = &self.path;
        let path_set: HashSet<usize> = path.clone().into_iter().collect();  // 最初の都市がない経路から作成

        // pathに対するLB範囲の出現数を算出
        let mut freq: Vec<Vec<usize>> = vec![vec![0; self.n]; self.n];
        for i in 0..path.len() {
            let t1 = path[i];
            for j in 1..2 {  // self.lb {
                if i+j == path.len() { break; }
                let t2 = path[i+j];
                freq[t1][t2] += 1;
                freq[t2][t1] += 1;
            }
        }

        // 優先度付きキューで、出現数が多い都市を順に配列Aに追加していく
        let mut heaps: Vec<BinaryHeap<(usize, usize)>> = vec![BinaryHeap::new(); self.n];
        for i in 0..self.n {
            for (j, cnt) in freq[i].iter().enumerate() {
                if i == j || cnt == &0 { continue; }
                heaps[i].push((*cnt, j));
            }
        }

        // 都市の道に沿って配列Aを設定
        let mut a: Vec<usize> = Vec::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        let mut next = 0;
        let mut now = next;
        while a.len() < self.n {
            now = next;
            a.push(now);
            visited[now] = true;
            let f = &freq[now];
            let mut max_freq = 0;
            let mut opt_v = 0;
            for (v, _) in self.path_freq[now].iter() {
                if !visited[*v] && f[*v] > max_freq {
                    max_freq = f[*v];
                    opt_v = *v;
                }
            }
            if max_freq > 0 {
                next = opt_v;
            }
            if next == now {
                for i in path_set.iter() {
                    if !visited[*i] {
                        next = *i;
                    }
                }
            }
        }
        // 入らなかったインデックスを入れる
        for i in path_set.iter() {
            if !visited[*i] {
                a.push(*i);
                next = *i;
            }
        }
        let mut prev = now;
        while a.len() < self.la {
            now = next;
            a.push(now);
            let list = self.path_freq.get(now).unwrap().keys().copied().collect::<Vec<usize>>();
            if list.len() > 1 {
                let i = self.rng.gen_range(0..list.len());
                next = list[i];
            }
            if list.len() == 1 || next == prev {
                next = self.rng.gen_range(0..self.n);
            }
            prev = now;
        }
        
        /*
        // 出現回数が大きいものから追加していくための配列
        let mut all_path_freq = self.all_path_freq.clone();

        // 出現回数が多いものを順に追加
        let mut a: Vec<usize> = Vec::new();
        let mut visited: Vec<bool> = vec![false; self.n];
        let mut now = 0;
        let mut max_c = 0;
        for (i, c) in all_path_freq.iter().enumerate() {
            if max_c < *c {
                max_c = *c;
                now = i;
            }
        }
        println!("# start now: {}, cnt: {}", now, max_c);
        all_path_freq[now] -= 1;
        let mut next = now;
        let mut cnt: usize;
        let mut on_lb: Vec<usize> = vec![0; self.lb];
        let mut lb_i = 1;
        // path_setに含まれる都市をすべて追加
        while a.len() < path_set.len() {
            now = next;
            a.push(now);
            visited[now] = true;
            all_path_freq[now] -= 1;
            on_lb[lb_i] = now;
            lb_i = (lb_i + 1) % self.lb;
            if !heaps[now].is_empty() {
                (cnt, next) = heaps[now].pop().unwrap();
                if !visited[next] {
                    continue;
                } else {
                    heaps[now].push((cnt, next));
                }
            }
            let mut max_c = 0;
            for (i, c) in all_path_freq.iter().enumerate() {
                if !visited[i] && max_c < *c {
                    next = i;
                    max_c = *c;
                }
            }
        }

        // 配列Aの残りを設定
        let mut all_heaps: BinaryHeap<(usize, usize)> = BinaryHeap::new();
        for (i, c) in all_path_freq.iter().enumerate() {
            all_heaps.push((*c, i));
        }
        add_cnt = 0;
        now = all_heaps.pop().unwrap().1;
        next = now;

        while a.len() < self.la {
            now = next;
            a.push(now);
            add_cnt += 1;
            all_path_freq[now] -= 1;
            if all_path_freq[now] > 0 {
                all_heaps.push((all_path_freq[now], now));
            }
            if !heaps[now].is_empty() && add_cnt < self.lb{
                (_, next) = heaps[now].pop().unwrap();
            }
            (_, next) = all_heaps.pop().unwrap();
        }*/

        println!("heaps: {:?}", heaps);
        for i in path_set.iter() {
            if !a.contains(i) {
                println!("# not found: {}", i);
            }
        }
        self.a = a;
    }

    fn optimize(&mut self, trial: usize) {
        println!("# optimize start");
        println!("# a: {:?}", self.a);
        let mut path: Vec<usize> = vec![0];  // 最初の都市を追加
        path.extend(&self.path);
        let path_set: HashSet<usize> = self.path.clone().into_iter().collect();  // 最初の都市がない経路から作成
        let path_list = path_set.iter().copied().collect::<Vec<usize>>();
        let mut match_cnt = self.match_cnt.clone();
        let mut eval_freq = self.eval_freq.clone();
        let mut pre_match_cnt = match_cnt.clone();
        let mut pre_eval_freq = eval_freq.clone();
        let mut pre_eval = self.eval;
        let mut a = self.a.clone();
        let mut must_a = self.must_a.clone();
        let mut skip_cnt = 0;
        let mut up_cnt = 0;
        let mut eq_cnt = 0;
        for _ in 0..trial {
            // a[i](ai) -> bi に変更
            let i = self.rng.gen_range(0..self.la);
            let ai = a[i];
            if must_a.contains_key(&ai) && must_a.get(&ai).unwrap() == &1 { skip_cnt+=1; continue; }  // 必須がなくなるなら処理しない
            let j = self.rng.gen_range(0..path_list.len());
            let b = path_list[j];
            self.change_a(i, &a, b, &mut match_cnt, &mut eval_freq);
            let mut eval:f64 = 1.0;
            for pi in path_set.iter() {
                eval *= eval_freq[*pi] as f64 / self.all_path_freq[*pi] as f64;
            }
            if eval >= pre_eval {
                if eval == pre_eval {
                    eq_cnt += 1;
                } else {
                    println!("# i: {}, ai => b: {} => {}", i, ai, b);
                    println!("# eval up: {} => {}", pre_eval, eval);
                    up_cnt += 1;
                }
                pre_eval = eval;
                pre_match_cnt = match_cnt.clone();
                pre_eval_freq = eval_freq.clone();
                if must_a.contains_key(&ai) {
                    println!("# must_a, ai: {}", ai);
                    *must_a.get_mut(&ai).unwrap() -= 1;
                    *must_a.get_mut(&b).unwrap() += 1;
                }
                a[i] = b;
            } else {
                match_cnt = pre_match_cnt.clone();
                eval_freq = pre_eval_freq.clone();
            }
        }
        println!("# trial: {}, up_cnt: {}, eq_cnt: {}, skip_cnt: {}", trial, up_cnt, eq_cnt, skip_cnt);
        self.a = a;
    }

    fn change_a(&mut self, i: usize, a: &Vec<usize>, b: usize, match_cnt: &mut Vec<HashMap<usize, usize>>, eval_freq: &mut Vec<usize>) {
        let ai = a[i];
        if ai == b { return; }
        let from = self.la + i - self.lb + 1;
        let to = self.la + i + self.lb - 1;
        for j in from..=to {
            let j = j % self.la;
            if i == j { continue; }  // 変更対象
            let cj = a[j];
            // aiを除外した処理
            if match_cnt[cj].contains_key(&ai) {
                let e = match_cnt[cj].entry(ai).or_insert(0);
                *e -= 1;
                if *e == 0 {
                    match_cnt[cj].remove(&ai);
                    eval_freq[cj] -= self.path_freq[cj].get(&ai).unwrap();
                }
            }
            if match_cnt[ai].contains_key(&cj) {
                let e = match_cnt[ai].entry(cj).or_insert(0);
                *e -= 1;
                if *e == 0 {
                    match_cnt[ai].remove(&cj);
                    eval_freq[ai] -= self.path_freq[ai].get(&cj).unwrap();
                }
            }
            // biを追加した処理
            if self.path_freq[cj].contains_key(&b) {
                let e = match_cnt[cj].entry(b).or_insert(0);
                *e += 1;
                if *e == 1 {
                    eval_freq[cj] += self.path_freq[cj].get(&b).unwrap();
                }
            }
            if self.path_freq[b].contains_key(&cj) {
                let e = match_cnt[b].entry(cj).or_insert(0);
                *e += 1;
                if *e == 1 {
                    eval_freq[b] += self.path_freq[b].get(&cj).unwrap();
                }
            }
        }
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
        let mut solver = setup();
        let pathes = solver.bfs(1, 3, 2);
        assert_eq!(pathes.len(), 2);
        assert_eq!(pathes[0], [0, 3]);
        assert_eq!(pathes[1], [0, 6, 5, 4, 3]);
        let pathes = solver.bfs(1, 3, 3);
        assert_eq!(pathes.len(), 2);
    }

    #[test]
    fn test_change_ai() {
        let (n, la, lb) = (10, 12, 3);
        let path = vec![1, 3, 0, 3, 4, 3, 0, 3, 4, 5, 6, 7, 6, 5, 4, 3, 8, 9, 8, 3];  // 最初の都市は含まれない
        let a = vec![0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 7, 6];
        assert_eq!(path.len(), 20);
        assert_eq!(a.len(), la);
        let mut a_opt = AOptimizer::new(n, la, lb);
        a_opt.init(&path);

        assert_eq!(a_opt.all_path_freq, [4, 2, 1, 6, 4, 3, 3, 2, 3, 2]);  // 初期値1
        let freq = &a_opt.path_freq;
        assert_eq!(freq[0].get(&1), Some(&1));
        assert_eq!(freq[0].get(&3), Some(&2));
        assert_eq!(freq[1].get(&3), Some(&1));
        assert_eq!(freq[1].get(&0), None);
        let mut match_cnt = a_opt.match_cnt.clone();
        assert_eq!(match_cnt[0].get(&1), Some(&1));
        assert_eq!(match_cnt[0].get(&2), None);
        assert_eq!(match_cnt[0].get(&3), None);
        assert_eq!(match_cnt[4].get(&5), None);
        assert_eq!(match_cnt[4].get(&3), Some(&1));
        assert_eq!(match_cnt[7].get(&6), Some(&3));
        let mut eval_freq = a_opt.eval_freq.clone();
        assert_eq!(a_opt.eval_freq, [2, 2, 1, 3, 3, 2, 3, 2, 2, 2]);  // 初期値1

        a_opt.change_a(2, &a, 3, &mut match_cnt, &mut eval_freq);
        assert_eq!(match_cnt[0].get(&1), Some(&1));
        assert_eq!(match_cnt[0].get(&3), Some(&1));
        assert_eq!(match_cnt[0].get(&2), None);
        assert_eq!(eval_freq[0], 4);
        assert_eq!(match_cnt[1].get(&3), Some(&2));
        assert_eq!(eval_freq[1], 2);
        assert_eq!(eval_freq[3], 5);
        assert_eq!(match_cnt[4].get(&3), Some(&2));
        assert_eq!(eval_freq[4], 3);

        a_opt.optimize(100);

        let path = vec![1, 3, 0, 3, 4, 3, 0, 3, 4, 5, 6, 7, 6, 5, 4, 3, 8, 9, 8, 3];  // 最初の都市は含まれない
        let a = vec![0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 7, 6];
        a_opt.init(&path);
        a_opt.change_a(7, &a, 0, &mut match_cnt, &mut eval_freq);
    }

    #[test]
    fn test_get_freq() {
        let (n, la, lb) = (7, 6, 3);
        let path = vec![1, 3, 1, 2, 4, 5, 1];  // 最初の都市は含まれない
        let a = vec![0, 1, 2, 3, 4, 5];
        let mut a_opt = AOptimizer::new(n, la, lb);
        a_opt.init(&path);
        a_opt.a = a;
        let match_rate = a_opt.match_rate();
        assert_eq!(match_rate, 14.0/18.0);

        a_opt.get_path_freq();  // 二度実行しても副作用なし(initで呼ばれている)
        let (freq, all_freq) = (&a_opt.path_freq, &a_opt.all_path_freq);
        println!("freq: {:?}", freq);
        println!("all_freq: {:?}", all_freq);
        assert_eq!(freq[0].get(&1), Some(&1));
        assert_eq!(freq[0].get(&0), None);
        assert_eq!(freq[0].get(&2), None);
        assert_eq!(all_freq[0], 2);
        assert_eq!(freq[1].get(&0), None);
        assert_eq!(freq[1].get(&1), None);
        assert_eq!(freq[1].get(&2), Some(&1));
        assert_eq!(freq[1].get(&3), Some(&1));
        assert_eq!(freq[1].get(&4), None);
        assert_eq!(freq[1].get(&5), None);
        assert_eq!(freq[5].get(&1), Some(&1));
        assert_eq!(all_freq[1], 3);
        assert_eq!(all_freq[6], 1);

        assert_eq!(a_opt.eval, 1.0);
        a_opt.optimize(10);
        assert_eq!(a_opt.eval, 1.0);
    }

}
