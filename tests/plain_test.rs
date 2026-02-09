use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray, new_ndarray};

#[test]
fn plain_value_lambda_format() {
    let format = format_new_ndarray!(for i => tensor[i, plain: 1]);

    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor). 0 ;
            let tsm_plain_0 : usize = 1 ;
            if tsm_plain_0 >= :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor). 1 {
                panic ! ("Plain value is out of bounds in tensor[_, plain]");
            };
            let tsm_ptr_tensor : * const _ = tensor . as_ptr ();
            let tsm_strides = tensor . strides ();
            let tsm_stride_0_tensor : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor : isize = tsm_strides [1usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 1usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor})});
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    (* unsafe {& * tsm_ptr_tensor . offset ((i as isize)* tsm_stride_0_tensor + (tsm_plain_0 as isize)* tsm_stride_1_tensor)})
                };
                unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );

    let format =
        format_new_ndarray!(for i j => tensor1[indexer[plain: 3 * n - 2, j], i, plain: 2 * n + 1]);

    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_2 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 0 ;
            if tsm_dim_2 != :: tensorism :: Reindexing2 :: get_output_bound (& indexer){
                panic ! ("Dimensions are not matching between tensor1[indexer, _, _] and indexer = indexer[_, _]");
            };
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 1 ;
            let tsm_dim_1 = :: tensorism :: Reindexing2 :: get_input1_bound (& indexer);
            let tsm_plain_0 : usize = 3 * n - 2 ;
            if tsm_plain_0 >= :: tensorism :: Reindexing2 :: get_input0_bound (& indexer){
                panic ! ("Plain value is out of bounds in indexer[plain, _]");
            };
            let tsm_plain_1 : usize = 2 * n + 1 ;
            if tsm_plain_1 >= :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 2 {
                panic ! ("Plain value is out of bounds in tensor1[_, _, plain]");
            };
            let tsm_ptr_tensor1 : * const _ = tensor1 . as_ptr ();
            let tsm_strides = tensor1 . strides ();
            let tsm_stride_0_tensor1 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor1 : isize = tsm_strides [1usize];
            let tsm_stride_2_tensor1 : isize = tsm_strides [2usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 2usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 , tsm_dim_1 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor1})});
            for i in 0usize .. tsm_dim_0 {
                for j in 0usize .. tsm_dim_1 {
                    let tsm_value = {
                        (* unsafe {& * tsm_ptr_tensor1 . offset ((:: tensorism :: Reindexing2 :: get_unchecked (& indexer , tsm_plain_0 , j)as isize)* tsm_stride_0_tensor1 + (i as isize)* tsm_stride_1_tensor1 + (tsm_plain_1 as isize)* tsm_stride_2_tensor1)})
                    };
                    unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
                };
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );
}

#[test]
fn plain_value_lambda_generation() {
    let tensor = Array2::<i32>::from_shape_fn((5, 3), |(i, j)| {
        ((i * j) as i32 - 3 * (i as i32) + 5 * (j as i32) - 6) % 11
    });
    let result = new_ndarray!(for i => tensor[i, plain: 1]);
    let expected = Array1::from_vec(vec![-1i32, -3, -5, -7, -9]);
    assert_eq!(expected, result);
}
