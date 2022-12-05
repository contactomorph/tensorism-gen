use crate::types::*;
use proc_macro2::{Delimiter, Group, Literal, Span, TokenStream, TokenTree};

fn sequentialize_tensor_func(func: RicciFunction, stream: &mut TokenStream) {
    let mut direct_indexes = func.inverted_indexes.clone();
    direct_indexes.reverse();
    let indexes_tuple = quote! {(#(#direct_indexes),*, )};
    let mut mappings = indexes_tuple.clone();
    for (i, index) in func.inverted_indexes.into_iter().enumerate() {
        let dimension_name = format_ident!("{}_dimension", index);
        let span = index.span();
        mappings = if i == 0 {
            quote_spanned! {span => (0usize..#dimension_name.into()).map(move |#index| #mappings)}
        } else {
            quote_spanned! {span => (0usize..#dimension_name.into()).flat_map(move |#index| #mappings)}
        }
    }
    let span = func.sequence.span;
    let mut content_stream = TokenStream::new();
    sequentialize_sequence(func.sequence, &mut content_stream);
    let func_stream = quote_spanned! {
        span =>  #mappings.map(|#indexes_tuple| { #content_stream })
    };
    stream.extend(func_stream);
}

fn sequentialize_sequence(sequence: RicciSequence, stream: &mut TokenStream) {
    let mut content = TokenStream::new();
    for alt in sequence.content.into_iter() {
        match alt {
            RicciAlternative::Func(sub_func) => {
                sequentialize_tensor_func(sub_func, &mut content);
            }
            RicciAlternative::Tree(token) => content.extend_one(token),
            RicciAlternative::Seq(sub_sequence) => {
                sequentialize_sequence(sub_sequence, &mut content);
            }
            RicciAlternative::TensorAccess {
                tensor_name,
                span,
                indexes,
            } => {
                let stream = quote_spanned! {
                    span => (* unsafe{ #tensor_name.get_unchecked(#(#indexes), *) })
                };
                content.extend(stream);
            }
        }
    }
    if sequence.use_parens {
        stream.extend_one(TokenTree::Group(Group::new(
            Delimiter::Parenthesis,
            content,
        )));
    } else {
        stream.extend(content)
    }
}

fn sequentialize_header(index_use: IndexUse) -> TokenStream {
    let mut output = TokenStream::new();
    for (name, positions) in index_use.into_iter() {
        let dimension_name = format_ident!("{}_dimension", name);
        let einstein_position = positions.first().unwrap();
        let pos = Literal::usize_unsuffixed(einstein_position.position);
        let tensor_name = einstein_position.tensor_name.clone();
        let length_definition = quote_spanned! {
            tensor_name.span() =>
            let #dimension_name: ::tensorism::dimensions::Dim::<_> = ::tensorism::tensors::Tensor::shape(&#tensor_name).#pos;
        };
        output.extend(length_definition);
        for einstein_position in positions.into_iter().skip(1) {
            let other_pos = Literal::usize_unsuffixed(einstein_position.position);
            let other_tensor_name = einstein_position.tensor_name.clone();
            let equality_assertion = quote_spanned! {
                einstein_position.index_name.span() =>
                :: tensorism::dimensions::identical(
                    ::tensorism::tensors::Tensor::shape(&#tensor_name).#pos,
                    ::tensorism::tensors::Tensor::shape(&#other_tensor_name).#other_pos
                );
            };
            output.extend(equality_assertion);
        }
    }
    output
}

fn try_extract_func(mut sequence: RicciSequence) -> Result<RicciFunction, RicciSequence> {
    if let [RicciAlternative::Func(_)] = sequence.content.as_slice() {
        if let Some(RicciAlternative::Func(func)) = sequence.content.pop() {
            if func.inverted_indexes.is_empty() {
                Err(func.sequence)
            } else {
                Ok(func)
            }
        } else {
            panic!("Unreachable")
        }
    } else {
        Err(sequence)
    }
}

fn sequentialize_shape_creation(mut func: RicciFunction) -> TokenStream {
    let mut direct_indexes = func.inverted_indexes.drain(..).collect::<Vec<_>>();
    direct_indexes.reverse();
    let index = direct_indexes.first().unwrap().clone();
    let dimension_name = format_ident!("{}_dimension", index);
    let mut shape_creation = quote_spanned! {
        index.span() => ::tensorism::building::TensorBuilding::with(#dimension_name)
    };
    for index in direct_indexes.iter().skip(1) {
        let dimension_name = format_ident!("{}_dimension", index);
        shape_creation = quote_spanned! {
            index.span() => #shape_creation.with(#dimension_name)
        }
    }
    let mut substream = TokenStream::new();
    sequentialize_sequence(func.sequence, &mut substream);
    quote_spanned! {
        Span::call_site() => #shape_creation.define(|(#(#direct_indexes),*, )| { #substream })
    }
}

fn sequentialize_body(sequence: RicciSequence, stream: &mut TokenStream) {
    match try_extract_func(sequence) {
        Ok(func) => stream.extend(sequentialize_shape_creation(func)),
        Err(sequence) => sequentialize_sequence(sequence, stream),
    }
}

pub fn sequentialize(sequence: RicciSequence, index_use: IndexUse) -> TokenStream {
    let mut stream = sequentialize_header(index_use);
    sequentialize_body(sequence, &mut stream);
    let mut output = TokenStream::new();
    output.extend_one(TokenTree::Group(Group::new(Delimiter::Brace, stream)));
    output.into()
}
