use proc_macro2::Ident;
use syn::Error;

use crate::analysis::types::{
    HeadKind, IndexingPosition, IndexingPositionContent, IndexingPositionMapping,
    InspectionCollector,
};
use crate::model::lambda::RicciLambda;
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

fn create_position(name: &Ident, position: usize, rank: usize, kind: HeadKind) -> IndexingPosition {
    match kind {
        HeadKind::Indexer => IndexingPosition {
            name: name.clone(),
            content: IndexingPositionContent::IndexerIndex(position, rank),
        },
        HeadKind::Tensor => IndexingPosition {
            name: name.clone(),
            content: IndexingPositionContent::TensorIndex(position, rank),
        },
    }
}

fn inspect_indexers(
    head_name: &Ident,
    position: usize,
    rank: usize,
    kind: HeadKind,
    indexer: &RicciIndexer,
    collector: &mut InspectionCollector,
) -> Result<(), syn::Error> {
    match indexer {
        RicciIndexer::Direct { index } => {
            let position = create_position(head_name, position, rank, kind);
            if !collector.try_add_position_to_existing_index(index, position) {
                create_unknown_index_error(index)?;
            }
        }
        RicciIndexer::Reverse { index } => {
            let position = create_position(head_name, position, rank, kind);
            if !collector.try_add_position_to_existing_index(index, position) {
                create_unknown_index_error(index)?;
            }
        }
        RicciIndexer::Reindexing {
            reindexing_name,
            indexers,
        } => {
            let position = create_position(head_name, position, rank, kind);
            collector.save_reindexing_result_equivalence(position, reindexing_name, rank);
            let rank = indexers.len();
            for (position, indexer) in indexers.iter().enumerate() {
                inspect_indexers(
                    reindexing_name,
                    position,
                    rank,
                    HeadKind::Indexer,
                    indexer,
                    collector,
                )?;
            }
        }
        RicciIndexer::Plain { expr } => {
            collector.add_plain_value(
                create_position(head_name, position, rank, kind),
                *expr.clone(),
            );
        }
    }
    Ok(())
}

fn inspect_lambda(
    lambda: &RicciLambda,
    collector: &mut InspectionCollector,
) -> Result<(), syn::Error> {
    let mut new_indexes = Vec::new();
    for index in &lambda.index_declaration.indexes {
        if !collector.try_declare_index(index.clone()) {
            create_duplicated_index_error(index)?;
        }
        new_indexes.push(index.clone());
    }
    for declaration in &lambda.alias_declarations {
        let index = &declaration.index;
        if !collector.try_declare_index(index.clone()) {
            create_duplicated_index_error(index)?;
        }
        match &declaration.indexer {
            RicciIndexer::Reindexing {
                reindexing_name,
                indexers,
            } => {
                let rank = indexers.len();
                for (position, indexer) in indexers.iter().enumerate() {
                    inspect_indexers(
                        reindexing_name,
                        position,
                        rank,
                        HeadKind::Indexer,
                        indexer,
                        collector,
                    )?;
                }
                let position = IndexingPosition {
                    name: reindexing_name.clone(),
                    content: IndexingPositionContent::IndexerResult(rank),
                };
                collector.try_add_position_to_existing_index(index, position);
            }
            _ => {
                todo!()
            }
        }
        new_indexes.push(index.clone());
    }
    inspect_segments(&lambda.body.segments, collector)?;
    for index in new_indexes {
        collector.save_existing_index(&index);
    }
    Ok(())
}

fn inspect_segments(
    segments: &[RicciSegment],
    collector: &mut InspectionCollector,
) -> Result<(), syn::Error> {
    for segment in segments {
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
                        collector,
                    )?;
                }
            }
            RicciSegment::SubLambda(lambda) => {
                inspect_lambda(lambda, collector)?;
            }
            RicciSegment::SubGroup { group, .. } => {
                inspect_segments(&group.segments, collector)?;
            }
            RicciSegment::Token(_) => {}
        }
    }
    Ok(())
}

fn inspect_main_group(
    group: &RicciGroup,
    collector: &mut InspectionCollector,
) -> Result<(), syn::Error> {
    if group.segments.len() == 1 && matches!(group.segments[0], RicciSegment::SubLambda(_)) {
        for segment in &group.segments {
            if let RicciSegment::SubLambda(lambda) = segment {
                if let Some(filter) = &lambda.filter {
                    return Err(Error::new_spanned(
                        filter.if_keyword,
                        "Macro level lambda cannot have a filter.",
                    ));
                }
                return inspect_lambda(lambda, collector);
            }
        }
        unreachable!()
    } else {
        inspect_segments(&group.segments, collector)
    }
}

pub fn inspect(group: &RicciGroup) -> Result<IndexingPositionMapping, syn::Error> {
    let mut collector = InspectionCollector::new();
    inspect_main_group(group, &mut collector)?;
    Ok(collector.into_mapping())
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
            ..
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
            ..
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

    #[test]
    fn inspect_invalid_lambda() {
        let tokens = quote!(for i if a[i] < 5 => a[i] + 3);

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        match inspect(&lambda) {
            Ok(_) => panic!("Expected error"),
            Err(err) => assert_eq!(err.to_string(), "Macro level lambda cannot have a filter."),
        }
    }
}
