use std::str::FromStr;

use tensorism::{building::TensorBuilder, shapes::ShapeBuilder, tensors::Tensor, dimensions::new_static_dim};
use tensorism_gen::{format_for_make, make};

#[test]
fn format_make_macro() {
    let string = format_for_make! {(i j $ a[i, j] + i as f64).sum()};
    assert_eq!(
        "{ \
            let i_dimension : :: tensorism :: dimensions :: Dim :: < _ > = :: tensorism :: tensors :: Tensor :: dims(& a).0 ; \
            let j_dimension : :: tensorism :: dimensions :: Dim :: < _ > = :: tensorism :: tensors :: Tensor :: dims(& a).1 ; \
            ((0usize .. i_dimension.into()).flat_map(move | i | (0usize .. j_dimension.into()).map(move | j | (i, j,)))\
            .map(| (i, j,) | { (* unsafe { a.get_unchecked(i, j) }) + i as f64 })).sum() \
        } ",
        string
    );
    let string = format_for_make! {i $ (j $ a[i, j] + b[j])};
    assert_eq!(
        "{ \
            let i_dimension : :: tensorism :: dimensions :: Dim :: < _ > = :: tensorism :: tensors :: Tensor :: dims(& a).0 ; \
            let j_dimension : :: tensorism :: dimensions :: Dim :: < _ > = :: tensorism :: tensors :: Tensor :: dims(& a).1 ; \
            :: tensorism :: dimensions :: identical(:: tensorism :: tensors :: Tensor :: dims(& a).1, :: tensorism :: tensors :: Tensor :: dims(& b).0) ; \
            :: tensorism :: shapes :: ShapeBuilder :: with(i_dimension).define(| (i,) | { \
                ((0usize .. j_dimension.into()).map(move | j | (j,))\
                .map(| (j,) | { (* unsafe { a.get_unchecked(i, j) }) + (* unsafe { b.get_unchecked(j) }) })) \
            }) \
        } ",
        string);
}

fn count_all_chars<'a>(it: impl Iterator<Item = &'a String>) -> usize {
    it.fold(0usize, |acc, message| acc + message.chars().count())
}

#[test]
fn run_make_macro() {
    let a = ShapeBuilder::with_static::<9>()
        .with_static::<10>()
        .define(|(i, j)| i as i64 * (j + 1) as i64);
    let sum: i64 = make! {(i j $ a[i, j] + i as i64).sum()};
    assert_eq!(2340i64, sum);

    let result: i64 = make! {Iterator::sum(i $ Iterator::min(j $ a[i, j]).unwrap())};
    assert_eq!(36i64, result);

    let messages = ["Hello", "World", "How", "are you?"].map(|s| String::from_str(s).unwrap());
    let c = ShapeBuilder::with_static::<4>()
        .prepare()
        .append_array(messages)
        .generate();
    let all_chars_count = make! {count_all_chars(i $ &c[i])};
    assert_eq!(21, all_chars_count);
    let b = ShapeBuilder::with_static::<10>().prepare().fill(&12f64);
    let t = make! {i j $ a[i, j] as f64 + b[j]};

    assert_eq!((new_static_dim::<9>(), new_static_dim::<10>()), t.dims());
}

//let v = tensorism_gen::decl!(i # a[i, 4] + b[i]);
//let v = {
//  if a.size1 != b.size1 { panic!("") };
//  (1..a.size1).build_iter_1(|i| a.get_raw(i, 4) + b.get_raw(i)).to_tensor_2();
//};

// #[fail]
// fn do_not_works() {
//     let value = tensorism_gen::decl!{
//         j $ i;
//     };
// }
