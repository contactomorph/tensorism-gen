use std::fmt::Display;

use proc_macro2::{Delimiter, TokenTree};
use syn::parse::{Parse, ParseStream};
use syn::{Error, Ident, Result, Token, braced, bracketed, parenthesized};

use crate::model::header::{RicciAliasDeclaration, RicciIndexDeclaration, RicciIndexer};

use phf::{Set, phf_set};

static ILLEGAL_KEYWORDS: Set<&'static str> = phf_set! {
    "let",
    "while",
    "loop",
    "break",
    "continue",
    "return",
    "fn",
    "struct",
    "enum",
    "type",
    "trait",
    "mod",
    "use",
    "extern",
};

pub enum RicciSegment {
    TensorCall {
        tensor_name: Ident,
        indexers: Vec<RicciIndexer>,
    },
    SubLambda(Box<RicciLambda>),
    SubGroup {
        delimiter: Delimiter,
        group: Box<RicciGroup>,
    },
    Token(TokenTree),
}

pub struct RicciGroup {
    pub segments: Vec<RicciSegment>,
}

pub struct RicciFilter {
    pub if_keyword: Token![if],
    pub segments: Vec<RicciSegment>,
}

pub struct RicciLambda {
    pub index_declaration: RicciIndexDeclaration,
    pub alias_declarations: Vec<RicciAliasDeclaration>,
    pub filter: Option<RicciFilter>,
    pub body: RicciGroup,
}

impl Parse for RicciSegment {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(syn::token::Paren) {
            let content;
            parenthesized!(content in input);

            let group: Box<RicciGroup> = content.parse()?;
            Ok(RicciSegment::SubGroup {
                group,
                delimiter: Delimiter::Parenthesis,
            })
        } else if input.peek(syn::token::Brace) {
            let content;
            braced!(content in input);

            let group: Box<RicciGroup> = content.parse()?;
            Ok(RicciSegment::SubGroup {
                group,
                delimiter: Delimiter::Brace,
            })
        } else if input.peek(syn::token::Bracket) {
            let content;
            bracketed!(content in input);

            let group: Box<RicciGroup> = content.parse()?;
            Ok(RicciSegment::SubGroup {
                group,
                delimiter: Delimiter::Bracket,
            })
        } else if input.peek(Token![for]) {
            let lambda: Box<RicciLambda> = input.parse()?;
            Ok(RicciSegment::SubLambda(lambda))
        } else if input.peek(Ident) && input.peek2(syn::token::Bracket) {
            let tensor_name: Ident = input.parse()?;
            let content;
            bracketed!(content in input);

            let mut indexers: Vec<RicciIndexer> = Vec::new();
            let mut first = true;
            while !content.is_empty() {
                if first {
                    first = false;
                } else {
                    content.parse::<Token![,]>()?;
                }
                let indexer: RicciIndexer = content.parse()?;
                indexers.push(indexer);
            }
            Ok(RicciSegment::TensorCall {
                tensor_name,
                indexers,
            })
        } else {
            let token: TokenTree = input.parse()?;
            if let TokenTree::Ident(ident) = &token {
                if ILLEGAL_KEYWORDS.contains(ident.to_string().as_str()) {
                    return Err(Error::new(
                        ident.span(),
                        format!("Keyword {} is illegal.", ident),
                    ));
                }
            }
            Ok(RicciSegment::Token(token))
        }
    }
}

impl Parse for RicciGroup {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut segments = Vec::<RicciSegment>::new();
        while !input.is_empty() {
            let segment = input.parse::<RicciSegment>()?;
            segments.push(segment);
        }
        Ok(Self { segments })
    }
}

impl Parse for RicciFilter {
    fn parse(input: ParseStream) -> Result<Self> {
        let if_keyword = input.parse::<Token![if]>()?;
        let mut segments = Vec::<RicciSegment>::new();
        while !input.is_empty() && !input.peek(Token![=>]) {
            let segment = input.parse::<RicciSegment>()?;
            segments.push(segment);
        }
        Ok(Self {
            if_keyword,
            segments,
        })
    }
}

impl Parse for RicciLambda {
    fn parse(input: ParseStream) -> Result<Self> {
        let index_declaration: RicciIndexDeclaration = input.parse()?;

        let mut alias_declarations: Vec<RicciAliasDeclaration> = Vec::new();

        while input.peek(Token![let]) {
            let alias_declaration: RicciAliasDeclaration = input.parse()?;
            alias_declarations.push(alias_declaration);
        }
        let filter = if input.peek(Token![if]) {
            Some(input.parse::<RicciFilter>()?)
        } else {
            None
        };

        input.parse::<Token![=>]>()?;
        let body: RicciGroup = input.parse()?;

        Ok(Self {
            index_declaration,
            alias_declarations,
            filter,
            body,
        })
    }
}

impl Display for RicciSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RicciSegment::TensorCall {
                tensor_name,
                indexers,
            } => {
                write!(f, " {} ⟦", tensor_name)?;
                let mut first = true;
                for indexer in indexers {
                    if !first {
                        f.write_str(" ,")?;
                    }
                    first = false;
                    indexer.fmt(f)?;
                }
                write!(f, " ⟧")
            }
            RicciSegment::SubLambda(lambda) => lambda.fmt(f),
            RicciSegment::SubGroup { delimiter, group } => match delimiter {
                Delimiter::Parenthesis => write!(f, " ({})", group),
                Delimiter::Brace => write!(f, " {{{}}}", group),
                Delimiter::Bracket => write!(f, " [{}]", group),
                Delimiter::None => group.fmt(f),
            },
            RicciSegment::Token(token) => {
                write!(f, " {}", token)
            }
        }
    }
}

impl Display for RicciGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for segment in &self.segments {
            segment.fmt(f)?;
        }
        Ok(())
    }
}

impl Display for RicciFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "?")?;
        for segment in &self.segments {
            segment.fmt(f)?;
        }
        Ok(())
    }
}

impl Display for RicciLambda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ", self.index_declaration)?;
        for alias_declaration in &self.alias_declarations {
            write!(f, "{} ", alias_declaration)?;
        }
        if let Some(filter) = &self.filter {
            write!(f, "{} ", filter)?;
        }
        write!(f, "▸{}", self.body)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::model::lambda::{RicciGroup, RicciLambda};

    use quote::quote;

    #[test]
    fn parse_lambda() {
        let tokens = quote!(for i => a[i] + 3);

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "∀ i ▸ a ⟦ i ⟧ + 3"
        );

        let tokens = quote!(for i j let k = sort[j] => a[i, j] + 4 * b[k]);

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "∀ i j ∙ k ≔ sort ⦇ j ⦈ ▸ a ⟦ i , j ⟧ + 4 * b ⟦ k ⟧"
        );
    }

    #[test]
    fn parse_complex() {
        let tokens = quote!(for i => a[i] + sum(for j => b[i, j]));

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "∀ i ▸ a ⟦ i ⟧ + sum (∀ j ▸ b ⟦ i , j ⟧)"
        );

        let tokens = quote!(for i => a[i] + sum(for j => b[every3[i], j]));

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "∀ i ▸ a ⟦ i ⟧ + sum (∀ j ▸ b ⟦ every3 ⦇ i ⦈ , j ⟧)"
        );

        let tokens = quote!(3.5 * median(for i => a[i] + sum(for j => b[every3[i], j])));

        assert_eq!(
            asserts::parse_and_display::<RicciGroup>(tokens),
            " 3.5 * median (∀ i ▸ a ⟦ i ⟧ + sum (∀ j ▸ b ⟦ every3 ⦇ i ⦈ , j ⟧))"
        );

        let tokens = quote!(3.5 * median(for i => a[i] + sum(for j => if i < j { b[every3[i], j] } else { c[j] + 4.0 })));

        assert_eq!(
            asserts::parse_and_display::<RicciGroup>(tokens),
            " 3.5 * median (∀ i ▸ a ⟦ i ⟧ + sum (∀ j ▸ if i < j { b ⟦ every3 ⦇ i ⦈ , j ⟧} else { c ⟦ j ⟧ + 4.0}))"
        );
    }

    #[test]
    fn parse_invalid() {
        let tokens = quote!(for i => { let name = a[i]; name });

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "Failed to parse type `tensorism_gen::model::lambda::RicciLambda`: Keyword let is illegal."
        );

        let tokens = quote!(for i => { while ok { a[i] } });

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "Failed to parse type `tensorism_gen::model::lambda::RicciLambda`: Keyword while is illegal."
        );
    }

    #[test]
    fn parse_filters() {
        let tokens = quote!(for i if b[i] < a[i] && 0.0 <= c[i, i] => a[i]);

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "∀ i ? b ⟦ i ⟧ < a ⟦ i ⟧ & & 0.0 < = c ⟦ i , i ⟧ ▸ a ⟦ i ⟧"
        );

        let tokens = quote!(for i => (for j if tensor2[j] < j => tensor1[i, j]).sum() + i);

        assert_eq!(
            asserts::parse_and_display::<RicciLambda>(tokens),
            "∀ i ▸ (∀ j ? tensor2 ⟦ j ⟧ < j ▸ tensor1 ⟦ i , j ⟧) . sum () + i"
        );
    }
}
