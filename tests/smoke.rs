#[test]
fn works() {
    // let value = tensorism_gen::decl!{
    //    i $ mean(j $ b[j, i] + c[i]);
    // };
    let value = tensorism_gen::decl!(j k l $ a[j, k] + i new_variable $ 3);
    assert_eq!(7, value);
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