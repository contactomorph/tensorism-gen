use std::collections::{HashMap, hash_map::Entry};

use proc_macro2::Ident;
use syn::Expr;

// A simple utility to generate increasing integers
pub struct IncreasingInteger {
    value: usize,
}

impl IncreasingInteger {
    pub fn new() -> Self {
        Self { value: 0 }
    }

    pub fn get_next(&mut self) -> usize {
        let current = self.value;
        self.value += 1;
        current
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum HeadKind {
    Tensor,
    Indexer,
}

// Represents a position where an index is used
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum IndexingPositionContent {
    // …[…, _, …]
    //      ^  ^
    //    pos  rank
    TensorIndex(usize, usize),
    // …[…, _, …]
    //      ^  ^
    //    pos  rank
    IndexerIndex(usize, usize),
    // …[…, …]
    // ^    ^
    //      rank
    IndexerResult(usize),
}

// Represents an occurrence of an index when calling a tensor or an indexer
// 〈name〉[…, _, …]
//           ^  ^
//         pos  rank
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct IndexingPosition {
    pub name: Ident,
    pub content: IndexingPositionContent,
}

#[cfg(test)]
impl PartialEq<(&str, usize, HeadKind)> for IndexingPosition {
    fn eq(&self, other: &(&str, usize, HeadKind)) -> bool {
        if self.name != other.0 {
            return false;
        }
        match self.content {
            IndexingPositionContent::IndexerResult(_) => {
                other.2 == HeadKind::Indexer && other.1 == usize::MAX
            }
            IndexingPositionContent::IndexerIndex(pos, _) => {
                other.2 == HeadKind::Indexer && other.1 == pos
            }
            IndexingPositionContent::TensorIndex(pos, _) => {
                other.2 == HeadKind::Tensor && other.1 == pos
            }
        }
    }
}

pub enum AliasSource {
    FromIndex { ricci_number: usize },
    FromReindexing { reindexing_name: Ident, rank: usize },
}

// Represents all occurrences of a specific index inside a Ricci lambda.
// ∀ 〈i〉 … ∙ … ≔ 〈reindexingA〉 ⦇ …, 〈i〉, … ⦈ … ▸ … 〈name1〉 ⟦ …, 〈i〉, … ⟧ … 〈nameN〉 ⟦ …, 〈i〉, … ⟧ …
// … ∙ 〈i〉 ≔ … ▸ … 〈name1〉 ⟦ …, 〈i〉, … ⟧ … 〈nameN〉 ⟦ …, 〈i〉, … ⟧ …
pub struct IndexingPositionEquivalence {
    // A Ricci number is just a unique integer attributed sequentially to each declared index
    // of a Ricci lambda. Instead of storing the index name, we store this number to avoid index name clashes
    // as the same index name can be used in different sub-lambdas.
    pub ricci_number: usize,
    pub index: Ident,
    pub alias_source: Option<AliasSource>,
    pub positions: Vec<IndexingPosition>,
}

impl IndexingPositionEquivalence {
    pub fn new(ricci_number: usize, index: Ident) -> Self {
        Self {
            ricci_number,
            index,
            alias_source: None,
            positions: Vec::new(),
        }
    }
    pub fn new_alias_declaration(
        ricci_number: usize,
        index: Ident,
        alias_source: AliasSource,
    ) -> Self {
        Self {
            ricci_number,
            index,
            alias_source: Some(alias_source),
            positions: Vec::new(),
        }
    }
}

pub struct PositionalPlainValue {
    pub plain_number: usize,
    pub position: IndexingPosition,
    pub expr: Expr,
}

// All data collected from inspecting indexing positions.
pub struct IndexingPositionMapping {
    // List of correspondences for all indexes found
    pub equivalences: Vec<IndexingPositionEquivalence>,
    // Plain values used as indexers: [ …, plain: 〈expr〉, …]
    pub plain_values: Vec<PositionalPlainValue>,
}

impl IndexingPositionMapping {
    pub fn new() -> Self {
        Self {
            equivalences: Vec::new(),
            plain_values: Vec::new(),
        }
    }
}

// A temporary collector only used to collect IndexingPosition data
pub struct InspectionCollector {
    free_ricci_number: IncreasingInteger,
    free_plain_number: IncreasingInteger,
    equivalences_per_index: HashMap<Ident, IndexingPositionEquivalence>,
    mapping: IndexingPositionMapping,
}

impl InspectionCollector {
    pub fn new() -> Self {
        Self {
            free_ricci_number: IncreasingInteger::new(),
            free_plain_number: IncreasingInteger::new(),
            equivalences_per_index: HashMap::new(),
            mapping: IndexingPositionMapping::new(),
        }
    }

    pub fn try_get_ricci_number(&self, index: &Ident) -> Option<usize> {
        self.equivalences_per_index
            .get(index)
            .map(|equivalence| equivalence.ricci_number)
    }

    pub fn try_declare_index(&mut self, index: Ident) -> bool {
        if let Entry::Vacant(entry) = self.equivalences_per_index.entry(index.clone()) {
            let ricci_number = self.free_ricci_number.get_next();
            let equivalence = IndexingPositionEquivalence::new(ricci_number, index);
            entry.insert(equivalence);
            true
        } else {
            false
        }
    }

    pub fn try_declare_alias(&mut self, index: Ident, alias_source: AliasSource) -> bool {
        if let Entry::Vacant(entry) = self.equivalences_per_index.entry(index.clone()) {
            let ricci_number = self.free_ricci_number.get_next();
            let equivalence = IndexingPositionEquivalence::new_alias_declaration(
                ricci_number,
                index,
                alias_source,
            );
            entry.insert(equivalence);
            true
        } else {
            false
        }
    }

    pub fn try_add_position_to_existing_index(
        &mut self,
        index: &Ident,
        position: IndexingPosition,
    ) -> bool {
        if let Some(equivalence) = self.equivalences_per_index.get_mut(index) {
            equivalence.positions.push(position);
            true
        } else {
            false
        }
    }

    pub fn add_plain_value(&mut self, position: IndexingPosition, expr: Expr) {
        let plain_number = self.free_plain_number.get_next();
        let plain_value = PositionalPlainValue {
            position,
            expr,
            plain_number,
        };
        self.mapping.plain_values.push(plain_value);
    }

    pub fn save_existing_index(&mut self, index: &Ident) {
        let equivalence = self
            .equivalences_per_index
            .remove(index)
            .unwrap_or_else(|| panic!("Equivalence not found for index {}", index));
        if !equivalence.positions.is_empty() {
            self.mapping.equivalences.push(equivalence);
        }
    }

    pub fn save_reindexing_result_equivalence(
        &mut self,
        position: IndexingPosition,
        reindexing_name: &Ident,
        rank: usize,
    ) {
        let ricci_number = self.free_ricci_number.get_next();
        let reindexing_position = IndexingPosition {
            name: reindexing_name.clone(),
            content: IndexingPositionContent::IndexerResult(rank),
        };
        let equivalence = IndexingPositionEquivalence {
            ricci_number,
            index: reindexing_name.clone(),
            alias_source: None,
            positions: vec![position, reindexing_position],
        };
        self.mapping.equivalences.push(equivalence);
    }

    pub fn into_mapping(self) -> IndexingPositionMapping {
        self.mapping
    }
}
