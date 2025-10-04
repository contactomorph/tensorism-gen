use std::collections::{HashMap, hash_map::Entry};

use proc_macro2::Ident;
use syn::Expr;

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

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum IndexingPositionContent {
    TensorIndex(usize, usize),
    IndexerIndex(usize, usize),
    IndexerResult,
}

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
            IndexingPositionContent::IndexerResult => {
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

pub struct IndexingPositionEquivalence {
    // A Ricci number is just a unique integer attributed sequentially to each declared index
    // of a Ricci lambda. Instead of storing the index name, we store this number to avoid index name clashes
    // as the same index name can be used in different sub-lambdas.
    pub ricci_number: usize,
    pub index: Ident,
    pub positions: Vec<IndexingPosition>,
}

impl IndexingPositionEquivalence {
    pub fn new(ricci_number: usize, index: Ident) -> Self {
        Self {
            ricci_number,
            index,
            positions: Vec::new(),
        }
    }
}

pub struct IndexingPositionMapping {
    pub equivalences: Vec<IndexingPositionEquivalence>,
    pub plain_values: HashMap<IndexingPosition, Expr>,
}

impl IndexingPositionMapping {
    pub fn new() -> Self {
        Self {
            equivalences: Vec::new(),
            plain_values: HashMap::new(),
        }
    }
}

pub struct InspectionCollector {
    free_ricci_number: IncreasingInteger,
    equivalences_per_index: HashMap<Ident, IndexingPositionEquivalence>,
    mapping: IndexingPositionMapping,
}

impl InspectionCollector {
    pub fn new() -> Self {
        Self {
            free_ricci_number: IncreasingInteger::new(),
            equivalences_per_index: HashMap::new(),
            mapping: IndexingPositionMapping::new(),
        }
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

    pub fn try_add_position_to_existing_index(
        &mut self,
        index: Ident,
        position: IndexingPosition,
    ) -> bool {
        if let Some(equivalence) = self.equivalences_per_index.get_mut(&index) {
            equivalence.positions.push(position);
            true
        } else {
            false
        }
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

    pub fn save_indexing_result_equivalence(
        &mut self,
        position: IndexingPosition,
        reindexing_name: &Ident,
    ) {
        let ricci_number = self.free_ricci_number.get_next();
        let reindexing_position = IndexingPosition {
            name: reindexing_name.clone(),
            content: IndexingPositionContent::IndexerResult,
        };
        let equivalence = IndexingPositionEquivalence {
            ricci_number,
            index: reindexing_name.clone(),
            positions: vec![position, reindexing_position],
        };
        self.mapping.equivalences.push(equivalence);
    }

    pub fn add_plain_value(&mut self, position: IndexingPosition, expr: Expr) {
        self.mapping.plain_values.insert(position, expr);
    }

    pub fn into_mapping(self) -> IndexingPositionMapping {
        self.mapping
    }
}
