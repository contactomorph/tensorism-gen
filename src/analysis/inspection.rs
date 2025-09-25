use std::collections::HashMap;

use proc_macro2::Ident;

use crate::analysis::types::{
    HeadKind, IndexingPosition, IndexingPositionEquivalence, IndexingPositionMapping,
};
use crate::model::{
    header::RicciIndexer,
    lambda::{RicciGroup, RicciSegment},
};

fn create_unknown_index_error(index: &Ident) -> Result<(), syn::Error> {
    Err(syn::Error::new_spanned(
        index,
        format!("Index '{}' is not declared in the current scope", index),
    ))
}

fn create_duplicated_index_error(index: &Ident) -> Result<(), syn::Error> {
    Err(syn::Error::new_spanned(
        index,
        format!("Index '{}' is already declared in the current scope", index),
    ))
}

const POSTFIX_CHARS: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j',
];

fn to_char(i: usize) -> char {
    *POSTFIX_CHARS.get(i).expect("Too many indexes!")
}

fn inspect_indexers(
    head_name: &Ident,
    position: usize,
    rank: usize,
    kind: HeadKind,
    indexer: &RicciIndexer,
    mapping: &mut IndexingPositionMapping,
    equivalences_per_index: &mut HashMap<Ident, IndexingPositionEquivalence>,
) -> Result<(), syn::Error> {
    match indexer {
        RicciIndexer::Direct { index } => {
            if !equivalences_per_index.contains_key(index) {
                create_unknown_index_error(index)?;
            }
            equivalences_per_index
                .entry(index.clone())
                .and_modify(|eq| {
                    eq.positions
                        .push(IndexingPosition::new(head_name, position, rank, kind))
                });
        }
        RicciIndexer::Reverse { index } => {
            if !equivalences_per_index.contains_key(index) {
                create_unknown_index_error(index)?;
            }
            equivalences_per_index
                .entry(index.clone())
                .and_modify(|eq| {
                    eq.positions
                        .push(IndexingPosition::new(head_name, position, rank, kind))
                });
        }
        RicciIndexer::Reindexing {
            reindexing_name,
            indexers,
        } => {
            let position_a = IndexingPosition::new(head_name, position, rank, kind);
            let position_b = IndexingPosition::new_indexer_result(reindexing_name);
            let equivalence = IndexingPositionEquivalence::from_positions(position_a, position_b);
            mapping.equivalences.push(equivalence);
            let rank = indexers.len();
            for (position, indexer) in indexers.iter().enumerate() {
                inspect_indexers(
                    reindexing_name,
                    position,
                    rank,
                    HeadKind::Indexer,
                    indexer,
                    mapping,
                    equivalences_per_index,
                )?;
            }
        }
        RicciIndexer::Plain { expr } => {
            mapping.plain_values.insert(
                IndexingPosition::new(head_name, position, rank, kind),
                *expr.clone(),
            );
        }
    }
    Ok(())
}

fn inspect_group(
    group: &RicciGroup,
    postfix: String,
    mapping: &mut IndexingPositionMapping,
    equivalences_per_index: &mut HashMap<Ident, IndexingPositionEquivalence>,
) -> Result<(), syn::Error> {
    for (i, segment) in group.segments.iter().enumerate() {
        match segment {
            RicciSegment::TensorCall {
                tensor_name,
                indexers,
            } => {
                let rank = indexers.len();
                for (position, indexer) in indexers.iter().enumerate() {
                    inspect_indexers(
                        tensor_name,
                        position,
                        rank,
                        HeadKind::Tensor,
                        indexer,
                        mapping,
                        equivalences_per_index,
                    )?;
                }
            }
            RicciSegment::SubLambda(lambda) => {
                let mut new_indexes = Vec::new();
                for index in &lambda.index_declaration.indexes {
                    if equivalences_per_index.contains_key(index) {
                        create_duplicated_index_error(index)?;
                    }
                    new_indexes.push(index.clone());
                    let equivalence =
                        IndexingPositionEquivalence::new(index.clone(), postfix.clone());
                    equivalences_per_index.insert(index.clone(), equivalence);
                }
                let new_postfix = format!("{}{}", postfix, to_char(i));
                inspect_group(&lambda.body, new_postfix, mapping, equivalences_per_index)?;
                for index in new_indexes {
                    let equivalence = equivalences_per_index.remove(&index).unwrap();
                    if !equivalence.positions.is_empty() {
                        mapping.equivalences.push(equivalence);
                    }
                }
            }
            RicciSegment::SubGroup { group, .. } => {
                let new_postfix = format!("{}{}", postfix, to_char(i));
                inspect_group(group, new_postfix, mapping, equivalences_per_index)?;
            }
            RicciSegment::Token(_) => {}
        }
    }
    Ok(())
}

pub fn inspect(group: &RicciGroup) -> Result<IndexingPositionMapping, syn::Error> {
    let postfix = String::new();
    let mut mapping = IndexingPositionMapping::new();
    let mut equivalences_per_index: HashMap<Ident, IndexingPositionEquivalence> = HashMap::new();
    inspect_group(group, postfix, &mut mapping, &mut equivalences_per_index)?;
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use crate::{
        analysis::{
            inspection::inspect,
            types::{HeadKind, IndexingPositionMapping},
        },
        model::lambda::RicciGroup,
    };

    use quote::quote;
    use syn::parse2;

    #[test]
    fn inspect_lambda() {
        let tokens = quote!(for i => a[i] + 3);

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping {
            equivalences,
            plain_values,
        } = mapping;

        assert_eq!(0, plain_values.len());
        assert_eq!(1, equivalences.len());
        assert_eq!(1, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("a", 0, HeadKind::Tensor));

        let tokens = quote!(for i j k => a[i, j] + b[j, k] * c[i]);

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping {
            equivalences,
            plain_values,
        } = mapping;

        assert_eq!(0, plain_values.len());
        assert_eq!(3, equivalences.len());
        assert_eq!(2, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("a", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[0].positions[1], &("c", 0, HeadKind::Tensor));
        assert_eq!(2, equivalences[1].positions.len());
        assert_eq!(&equivalences[1].positions[0], &("a", 1, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[1], &("b", 0, HeadKind::Tensor));
        assert_eq!(1, equivalences[2].positions.len());
        assert_eq!(&equivalences[2].positions[0], &("b", 1, HeadKind::Tensor));
    }

    #[test]
    fn inspect_group() {
        let tokens = quote!(product(for i => a[i] + sum(for j => b[i, j]) * mean(for j => c[j, i, j] / d[j])));

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping { equivalences, .. } = mapping;

        assert_eq!(3, equivalences.len());
        assert_eq!(1, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("b", 1, HeadKind::Tensor));
        assert_eq!(3, equivalences[1].positions.len());
        assert_eq!(&equivalences[1].positions[0], &("c", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[1], &("c", 2, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[2], &("d", 0, HeadKind::Tensor));
        assert_eq!(3, equivalences[2].positions.len());
        assert_eq!(&equivalences[2].positions[0], &("a", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[2].positions[1], &("b", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[2].positions[2], &("c", 1, HeadKind::Tensor));

        let tokens = quote!(3.5 * median(
            for i => a[i] +
                sum(for j => if i < j { b[every3[i], j] } else { c[j] + 4.0 }) -
                max(for j => d[j, every4[j]] - e[sorted_by_a[j], i])
        ));

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping { equivalences, .. } = mapping;

        assert_eq!(6, equivalences.len());
        assert_eq!(2, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("b", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[0].positions[1],
            &("every3", usize::MAX, HeadKind::Indexer)
        );
        assert_eq!(2, equivalences[1].positions.len());
        assert_eq!(&equivalences[1].positions[0], &("b", 1, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[1], &("c", 0, HeadKind::Tensor));
        assert_eq!(2, equivalences[2].positions.len());
        assert_eq!(&equivalences[2].positions[0], &("d", 1, HeadKind::Tensor));
        assert_eq!(
            &equivalences[2].positions[1],
            &("every4", usize::MAX, HeadKind::Indexer)
        );
        assert_eq!(2, equivalences[3].positions.len());
        assert_eq!(&equivalences[3].positions[0], &("e", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[3].positions[1],
            &("sorted_by_a", usize::MAX, HeadKind::Indexer)
        );
        assert_eq!(3, equivalences[4].positions.len());
        assert_eq!(&equivalences[4].positions[0], &("d", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[4].positions[1],
            &("every4", 0, HeadKind::Indexer)
        );
        assert_eq!(
            &equivalences[4].positions[2],
            &("sorted_by_a", 0, HeadKind::Indexer)
        );
        assert_eq!(3, equivalences[5].positions.len());
        assert_eq!(&equivalences[5].positions[0], &("a", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[5].positions[1],
            &("every3", 0, HeadKind::Indexer)
        );
        assert_eq!(&equivalences[5].positions[2], &("e", 1, HeadKind::Tensor));
    }
}
