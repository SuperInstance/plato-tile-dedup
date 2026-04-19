//! plato-tile-dedup — Tile deduplication
//!
//! Find exact and near-duplicate tiles. Merge them. Clean up the warehouse.
//!
//! ```rust
//! let tiles = vec![tile("t1", "What is 2+2", "4"), tile("t2", "What is 2+2", "Four")];
//! let exact = find_exact_duplicates(&tiles);
//! let near = find_near_duplicates(&tiles, 0.5);
//! let merged = merge_tiles(&[&tiles[0], &tiles[1]]);
//! ```

#[derive(Debug, Clone)]
pub struct DedupTile {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub domain: String,
    pub confidence: f64,
    pub tags: Vec<String>,
}

impl DedupTile {
    pub fn new(id: &str, q: &str, a: &str, domain: &str, conf: f64) -> Self {
        Self { id: id.to_string(), question: q.to_string(), answer: a.to_string(),
               domain: domain.to_string(), confidence: conf, tags: Vec::new() }
    }

    pub fn words(&self) -> Vec<String> {
        let mut all = self.question.clone();
        all.push(' ');
        all.push_str(&self.answer);
        all.split_whitespace().map(|w| w.to_lowercase()).collect()
    }
}

/// Find groups of exact duplicates (same question text, case-insensitive).
pub fn find_exact_duplicates(tiles: &[DedupTile]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; tiles.len()];

    for i in 0..tiles.len() {
        if assigned[i] { continue; }
        let mut group = vec![i];
        assigned[i] = true;
        let qi = tiles[i].question.to_lowercase();
        for j in (i+1)..tiles.len() {
            if assigned[j] { continue; }
            if tiles[j].question.to_lowercase() == qi {
                group.push(j);
                assigned[j] = true;
            }
        }
        if group.len() > 1 { groups.push(group); }
    }
    groups
}

/// Jaccard word overlap between two word sets.
pub fn jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() { return 1.0; }
    if a.is_empty() || b.is_empty() { return 0.0; }
    let set_a: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    intersection as f64 / union as f64
}

/// Find groups of near-duplicates (Jaccard >= threshold).
pub fn find_near_duplicates(tiles: &[DedupTile], threshold: f64) -> Vec<Vec<usize>> {
    let words: Vec<Vec<String>> = tiles.iter().map(|t| t.words()).collect();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; tiles.len()];

    for i in 0..tiles.len() {
        if assigned[i] { continue; }
        let mut group = vec![i];
        assigned[i] = true;
        for j in (i+1)..tiles.len() {
            if assigned[j] { continue; }
            if jaccard(&words[i], &words[j]) >= threshold {
                group.push(j);
                assigned[j] = true;
            }
        }
        if group.len() > 1 { groups.push(group); }
    }
    groups
}

/// Merge multiple tiles into one (best answer, max confidence, union tags).
pub fn merge_tiles(tiles: &[&DedupTile]) -> DedupTile {
    assert!(!tiles.is_empty());
    if tiles.len() == 1 { return tiles[0].clone(); }

    let best = tiles.iter().max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap()).unwrap();
    let mut merged_tags: Vec<String> = Vec::new();
    for t in tiles {
        for tag in &t.tags {
            if !merged_tags.contains(tag) { merged_tags.push(tag.clone()); }
        }
    }

    let mut merged = DedupTile::new(
        &tiles[0].id, // keep first ID
        &tiles[0].question,
        &best.answer,
        &best.domain,
        best.confidence,
    );
    merged.tags = merged_tags;

    // Merge answers if different
    let answers: Vec<&str> = tiles.iter().map(|t| t.answer.as_str()).collect();
    let unique_answers: std::collections::HashSet<&str> = answers.iter().cloned().collect();
    if unique_answers.len() > 1 {
        merged.answer = tiles.iter()
            .filter(|t| t.confidence >= best.confidence - 0.1)
            .map(|t| t.answer.as_str())
            .collect::<Vec<&str>>()
            .join(" | ");
    }

    merged
}

/// Deduplicate a tile vector in-place. Returns count removed.
pub fn dedup_store(tiles: &mut Vec<DedupTile>, threshold: f64) -> usize {
    let dup_groups = find_near_duplicates(tiles, threshold);
    let mut to_remove = std::collections::HashSet::new();

    for group in &dup_groups {
        let merged = merge_tiles(&group.iter().map(|&i| &tiles[i]).collect::<Vec<_>>());
        // Keep first, mark rest for removal
        for &idx in &group[1..] {
            to_remove.insert(idx);
        }
        // Update the kept tile
        if let Some(keep_idx) = group.first() {
            tiles[*keep_idx] = merged;
        }
    }

    let original_len = tiles.len();
    let mut new_tiles = Vec::new();
    for (i, tile) in tiles.drain(..).enumerate() {
        if !to_remove.contains(&i) { new_tiles.push(tile); }
    }
    *tiles = new_tiles;
    original_len - tiles.len()
}

/// Find tile IDs that are suspect duplicates.
pub fn find_duplicate_ids(tiles: &[DedupTile], threshold: f64) -> Vec<(String, String, f64)> {
    let mut pairs = Vec::new();
    let words: Vec<Vec<String>> = tiles.iter().map(|t| t.words()).collect();
    for i in 0..tiles.len() {
        for j in (i+1)..tiles.len() {
            let sim = jaccard(&words[i], &words[j]);
            if sim >= threshold {
                pairs.push((tiles[i].id.clone(), tiles[j].id.clone(), sim));
            }
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(id: &str, q: &str, a: &str, domain: &str, conf: f64) -> DedupTile {
        DedupTile::new(id, q, a, domain, conf)
    }

    #[test]
    fn test_exact_duplicates_found() {
        let tiles = vec![
            t("t1", "What is 2+2", "4", "math", 0.9),
            t("t2", "What is 2+2", "Four", "math", 0.8),
        ];
        let groups = find_exact_duplicates(&tiles);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn test_exact_duplicates_case_insensitive() {
        let tiles = vec![
            t("t1", "What is 2+2", "4", "math", 0.9),
            t("t2", "what is 2+2", "four", "math", 0.8),
        ];
        let groups = find_exact_duplicates(&tiles);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_no_exact_duplicates() {
        let tiles = vec![
            t("t1", "What is 2+2", "4", "math", 0.9),
            t("t2", "What is 3+3", "6", "math", 0.9),
        ];
        assert!(find_exact_duplicates(&tiles).is_empty());
    }

    #[test]
    fn test_near_duplicates_found() {
        let tiles = vec![
            t("t1", "Pythagorean theorem formula", "a²+b²=c²", "math", 0.9),
            t("t2", "The Pythagorean theorem is", "a²+b²=c²", "math", 0.8),
        ];
        let groups = find_near_duplicates(&tiles, 0.3);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_jaccard_identical() {
        let a = vec!["hello".to_string(), "world".to_string()];
        let b = vec!["world".to_string(), "hello".to_string()];
        assert!((jaccard(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = vec!["hello".to_string()];
        let b = vec!["world".to_string()];
        assert!((jaccard(&a, &b)).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_empty() {
        assert!((jaccard(&[], &[]) - 1.0).abs() < 0.001);
        assert!((jaccard(&["x".to_string()], &[])).abs() < 0.001);
    }

    #[test]
    fn test_merge_tiles() {
        let tiles = vec![
            t("t1", "Q", "low conf answer", "d", 0.5),
            t("t2", "Q", "high conf answer", "d", 0.95),
        ];
        let merged = merge_tiles(&tiles.iter().collect::<Vec<_>>());
        assert_eq!(merged.answer, "high conf answer");
        assert!((merged.confidence - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_merge_with_tags() {
        let mut t1 = t("t1", "Q", "A", "d", 0.9);
        t1.tags = vec!["math".to_string()];
        let mut t2 = t("t2", "Q", "B", "d", 0.8);
        t2.tags = vec!["geometry".to_string()];
        let merged = merge_tiles(&vec![&t1, &t2]);
        assert_eq!(merged.tags.len(), 2);
    }

    #[test]
    fn test_merge_single() {
        let tile = t("t1", "Q", "A", "d", 0.9);
        let merged = merge_tiles(&vec![&tile]);
        assert_eq!(merged.id, "t1");
    }

    #[test]
    fn test_dedup_store() {
        let mut tiles = vec![
            t("t1", "What is 2+2", "4", "math", 0.9),
            t("t2", "What is 2+2", "Four", "math", 0.8),
            t("t3", "What is pi", "3.14", "math", 0.9),
        ];
        let removed = dedup_store(&mut tiles, 0.5);
        assert_eq!(removed, 1);
        assert_eq!(tiles.len(), 2);
    }

    #[test]
    fn test_dedup_no_dups() {
        let mut tiles = vec![
            t("t1", "math question", "answer", "math", 0.9),
            t("t2", "cooking recipe", "flour", "cook", 0.9),
        ];
        assert_eq!(dedup_store(&mut tiles, 0.5), 0);
    }

    #[test]
    fn test_find_duplicate_ids() {
        let tiles = vec![
            t("t1", "what is math", "numbers", "math", 0.9),
            t("t2", "math is what", "numbers", "math", 0.8),
            t("t3", "baking bread", "flour", "cook", 0.9),
        ];
        let pairs = find_duplicate_ids(&tiles, 0.3);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "t1");
        assert_eq!(pairs[0].1, "t2");
    }

    #[test]
    fn test_empty_tiles() {
        assert!(find_exact_duplicates(&[]).is_empty());
        assert!(find_near_duplicates(&[], 0.5).is_empty());
        assert_eq!(dedup_store(&mut Vec::new(), 0.5), 0);
    }

    #[test]
    fn test_words_extraction() {
        let tile = t("t1", "Hello World", "test answer", "d", 0.5);
        let words = tile.words();
        assert!(words.contains(&"hello".to_string()));
        assert!(words.contains(&"world".to_string()));
        assert!(words.contains(&"test".to_string()));
    }
}
