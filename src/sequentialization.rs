use crate::types::*;
use proc_macro2::{Delimiter, Group, Ident, Literal, Span, TokenStream, TokenTree};

fn sequentialize_parens_group(span: Span, group: Vec<EinsteinAlternative>) -> TokenTree {
    let inner_func = EinsteinFunction::from_group(span, group);
    let mut inner_stream = TokenStream::new();
    sequentialize_body(inner_func, &mut inner_stream);
    TokenTree::Group(Group::new(Delimiter::Parenthesis, inner_stream))
}

fn sequentialize_tensor_func(
    span: Span,
    inverted_indexes: Vec<Ident>,
    stream: TokenStream,
) -> TokenTree {
    let index_count = inverted_indexes.len();
    let func_stream = if index_count > 0 {
        let mut direct_indexes = inverted_indexes.clone();
        direct_indexes.reverse();
        let indexes_tuple = quote! {(#(#direct_indexes),*, )};
        let mut content = indexes_tuple.clone();
        for (i, index) in inverted_indexes.into_iter().enumerate() {
            let length_name = format_ident!("{}_length", index);
            let span = index.span();
            content = if i == 0 {
                quote_spanned! {span => (0usize..#length_name).map(move |#index| #content)}
            } else {
                quote_spanned! {span => (0usize..#length_name).flat_map(move |#index| #content)}
            }
        }
        quote_spanned! {span =>  #content.map(|#indexes_tuple| { #stream }) }
    } else {
        stream
    };
    TokenTree::Group(Group::new(Delimiter::None, func_stream))
}

fn sequentialize_body(func: EinsteinFunction, stream: &mut TokenStream) {
    let mut content = TokenStream::new();
    for alt in func.content.into_iter() {
        match alt {
            EinsteinAlternative::Func(sub_func) => {
                sequentialize_body(sub_func, &mut content);
            }
            EinsteinAlternative::Tree(token) => content.extend_one(token),
            EinsteinAlternative::ParensGroup(span, group) => {
                content.extend_one(sequentialize_parens_group(span, group));
            }
            EinsteinAlternative::TensorAccess {
                tensor_name,
                span,
                indexes,
            } => {
                let stream = quote_spanned! {span => (* unsafe{ #tensor_name.get_unchecked(#(#indexes), *) })};
                content.extend(stream);
            }
        }
    }
    stream.extend_one(sequentialize_tensor_func(
        func.span,
        func.inverted_indexes,
        content,
    ));
}

fn sequentialize_header(index_use: IndexUse) -> TokenStream {
    let mut output = TokenStream::new();
    for (name, positions) in index_use.into_iter() {
        let length_name = format_ident!("{}_length", name);
        let einstein_position = positions.first().unwrap();
        let pos = Literal::usize_unsuffixed(einstein_position.position);
        let tensor_name = einstein_position.tensor_name.clone();
        let length_definition = quote! {let #length_name: usize = ::tensorism::tensors::Tensor::dims(&#tensor_name).#pos.into();};
        output.extend(length_definition);
    }
    output
}

pub fn sequentialize(func: EinsteinFunction, index_use: IndexUse) -> TokenStream {
    let mut stream = sequentialize_header(index_use);
    sequentialize_body(func, &mut stream);
    let mut output = TokenStream::new();
    output.extend_one(TokenTree::Group(Group::new(Delimiter::Brace, stream)));
    output.into()
}
