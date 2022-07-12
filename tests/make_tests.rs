use std::str::FromStr;

use tensorism::{building::TensorBuilder, shapes::ShapeBuilder};
use tensorism_gen::{make, format_for_make};

#[test]
fn format_make_macro() {
    let string = format_for_make! {(i j $ a[i, j] + i as f64).sum()};
    assert_eq!(
        "{ \
            let i_length : usize = :: tensorism :: tensors :: Tensor :: dims(& a).0.into() ; \
            let j_length : usize = :: tensorism :: tensors :: Tensor :: dims(& a).1.into() ; \
            ((0usize .. i_length).flat_map(move | i | (0usize .. j_length).map(move | j | (i, j,)))\
            .map(| (i, j,) | { (* unsafe { a.get_unchecked(i, j) }) + i as f64 })).sum() \
        } ",
        string
    );
    let string = format_for_make! {i $ (j $ a[i, j] + b[j])};
    assert_eq!(
        "{ \
            let i_length : usize = :: tensorism :: tensors :: Tensor :: dims(& a).0.into() ; \
            let j_length : usize = :: tensorism :: tensors :: Tensor :: dims(& a).1.into() ; \
            (0usize .. i_length).map(move | i | (i,)).map(| (i,) | { (\
                (0usize .. j_length).map(move | j | (j,))\
                .map(| (j,) | { (* unsafe { a.get_unchecked(i, j) }) + (* unsafe { b.get_unchecked(j) }) })\
            ) }) \
        } ",
        string);
}

fn count_all_chars<'a>(it: impl Iterator<Item = &'a String>) -> usize {
    it.fold(0usize, |acc, message| acc + message.chars().count())
}

#[test]
fn run_make_macro() {
    let a = ShapeBuilder::with_static::<10>()
        .with_first()
        .define(|(i, j)| i as i64 * (j + 1) as i64);
    let sum: i64 = make! {(i j $ a[i, j] + i as i64).sum()};
    assert_eq!(2925i64, sum);

    let result: i64 = make! {Iterator::sum(i $ Iterator::min(j $ a[i, j]).unwrap())};
    assert_eq!(45i64, result);

    let messages = ["Hello", "World", "How", "are you?"].map(|s| String::from_str(s).unwrap());
    let c = ShapeBuilder::with_static::<4>()
        .prepare()
        .append_array(messages)
        .generate();
    let all_chars_count = make! {count_all_chars(i $ &c[i])};
    assert_eq!(21, all_chars_count);
    // let b = ShapeBuilder::with_static::<10>()
    //     .prepare()
    //     .fill(&12f64);
    // let iterator = make! {i $ (j $ a[i, j] + b[j])};
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
