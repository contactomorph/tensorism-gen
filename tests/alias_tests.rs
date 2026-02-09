use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray, new_ndarray};

#[test]
fn indexing_alias_format() {
    let format = format_new_ndarray!(for i let j = sort[i] => tensor1[j]);

    asserts::equivalent!(
        format,
        r##"{
            let tsm_dim_0 = :: tensorism :: Reindexing1 :: get_input0_bound(& sort);
            let tsm_dim_1 = :: tensorism :: Reindexing1 :: get_output_bound(& sort);
            if tsm_dim_1 != :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor1) {
                panic! ("Dimensions are not matching between j = sort[_] and tensor1[j]");
            };
            let tsm_ptr_tensor1 : * const _ = tensor1.as_ptr();
            let tsm_strides = tensor1.strides();
            let tsm_stride_0_tensor1 : isize = tsm_strides [0usize];
            type TsmDimensionType = :: ndarray :: Dim < [ :: ndarray :: Ix; 1usize ] > ;
            let mut tsm_res = :: ndarray :: Array :: < _, TsmDimensionType > :: uninit((tsm_dim_0,));
            let mut tsm_res_ptr = tsm_res.as_mut_ptr() as * mut _;
            fn tsm_unify < T, D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T, _f : impl Fn() -> T, ) {}
            tsm_unify(& tsm_res, tsm_res_ptr, || { ( * unsafe { & * tsm_ptr_tensor1 } ) });
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    let j = unsafe {:: tensorism :: Reindexing1 :: get_unchecked(& sort, i)};
                    ( * unsafe { & * tsm_ptr_tensor1.offset((j as isize) * tsm_stride_0_tensor1)} )
                };
                unsafe {tsm_res_ptr.write(tsm_value); tsm_res_ptr = tsm_res_ptr.add(1);}
            };
            unsafe {tsm_res.assume_init()}
        } "##
    );

    let format = format_new_ndarray!(for i j let k = combine[i, j] => tensor2[i, k]);
    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_0 = :: tensorism :: Reindexing2 :: get_input0_bound (& combine);
            if tsm_dim_0 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor2). 0 {
                panic ! ("Dimensions are not matching between combine[i, _] and tensor2[i, _]");
            };
            let tsm_dim_1 = :: tensorism :: Reindexing2 :: get_input1_bound (& combine);
            let tsm_dim_2 = :: tensorism :: Reindexing2 :: get_output_bound (& combine);
            if tsm_dim_2 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor2). 1 {
                panic ! ("Dimensions are not matching between k = combine[_, _] and tensor2[_, k]");
            };
            let tsm_ptr_tensor2 : * const _ = tensor2 . as_ptr ();
            let tsm_strides = tensor2 . strides ();
            let tsm_stride_0_tensor2 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor2 : isize = tsm_strides [1usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 2usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 , tsm_dim_1 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor2})});
            for i in 0usize .. tsm_dim_0 {
                for j in 0usize .. tsm_dim_1 {
                    let tsm_value = {
                        let k = unsafe {:: tensorism :: Reindexing2 :: get_unchecked (& combine , i , j)};
                        (* unsafe {& * tsm_ptr_tensor2 . offset ((i as isize)* tsm_stride_0_tensor2 + (k as isize)* tsm_stride_1_tensor2)})
                    };
                    unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
                };
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );

    let format = format_new_ndarray!(for i j let k = sup[middle[sub[i, j], j]] => tensor3[k, j, i]);
    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_2 = :: tensorism :: Reindexing1 :: get_input0_bound (& sup);
            if tsm_dim_2 != :: tensorism :: Reindexing2 :: get_output_bound (& middle){
                panic ! ("Dimensions are not matching between sup[middle] and middle = middle[_, _]");
            };
            let tsm_dim_3 = :: tensorism :: Reindexing2 :: get_input0_bound (& middle);
            if tsm_dim_3 != :: tensorism :: Reindexing2 :: get_output_bound (& sub){
                panic ! ("Dimensions are not matching between middle[sub, _] and sub = sub[_, _]");
            };
            let tsm_dim_0 = :: tensorism :: Reindexing2 :: get_input0_bound (& sub);
            if tsm_dim_0 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3). 2 {
                panic ! ("Dimensions are not matching between sub[i, _] and tensor3[_, _, i]");
            };
            let tsm_dim_1 = :: tensorism :: Reindexing2 :: get_input1_bound (& sub);
            if tsm_dim_1 != :: tensorism :: Reindexing2 :: get_input1_bound (& middle){
                panic ! ("Dimensions are not matching between sub[_, j] and middle[_, j]");
            };
            if tsm_dim_1 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3). 1 {
                panic ! ("Dimensions are not matching between sub[_, j] and tensor3[_, j, _]");
            };
            let tsm_dim_4 = :: tensorism :: Reindexing1 :: get_output_bound (& sup);
            if tsm_dim_4 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3). 0 {
                panic ! ("Dimensions are not matching between k = sup[_] and tensor3[k, _, _]");
            };
            let tsm_ptr_tensor3 : * const _ = tensor3 . as_ptr ();
            let tsm_strides = tensor3 . strides ();
            let tsm_stride_0_tensor3 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor3 : isize = tsm_strides [1usize];
            let tsm_stride_2_tensor3 : isize = tsm_strides [2usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 2usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 , tsm_dim_1 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor3})});
            for i in 0usize .. tsm_dim_0 {
                for j in 0usize .. tsm_dim_1 {
                    let tsm_value = {
                        let k = unsafe {:: tensorism :: Reindexing1 :: get_unchecked (& sup , :: tensorism :: Reindexing2 :: get_unchecked (& middle , :: tensorism :: Reindexing2 :: get_unchecked (& sub , i , j), j))};
                        (* unsafe {& * tsm_ptr_tensor3 . offset ((k as isize)* tsm_stride_0_tensor3 + (j as isize)* tsm_stride_1_tensor3 + (i as isize)* tsm_stride_2_tensor3)})
                    };
                    unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
                };
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );

    let format = format_new_ndarray!(for i j => tensor3[sup[middle[sub[i, j], j]], j, i]);
    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_2 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3). 0 ;
            if tsm_dim_2 != :: tensorism :: Reindexing1 :: get_output_bound (& sup){
                panic ! ("Dimensions are not matching between tensor3[sup, _, _] and sup = sup[_]");
            };
            let tsm_dim_3 = :: tensorism :: Reindexing1 :: get_input0_bound (& sup);
            if tsm_dim_3 != :: tensorism :: Reindexing2 :: get_output_bound (& middle){
                panic ! ("Dimensions are not matching between sup[middle] and middle = middle[_, _]");
            };
            let tsm_dim_4 = :: tensorism :: Reindexing2 :: get_input0_bound (& middle);
            if tsm_dim_4 != :: tensorism :: Reindexing2 :: get_output_bound (& sub){
                panic ! ("Dimensions are not matching between middle[sub, _] and sub = sub[_, _]");
            };
            let tsm_dim_0 = :: tensorism :: Reindexing2 :: get_input0_bound (& sub);
            if tsm_dim_0 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3). 2 {
                panic ! ("Dimensions are not matching between sub[i, _] and tensor3[_, _, i]");
            };
            let tsm_dim_1 = :: tensorism :: Reindexing2 :: get_input1_bound (& sub);
            if tsm_dim_1 != :: tensorism :: Reindexing2 :: get_input1_bound (& middle){
                panic ! ("Dimensions are not matching between sub[_, j] and middle[_, j]");
            };
            if tsm_dim_1 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3). 1 {
                panic ! ("Dimensions are not matching between sub[_, j] and tensor3[_, j, _]");
            };
            let tsm_ptr_tensor3 : * const _ = tensor3 . as_ptr ();
            let tsm_strides = tensor3 . strides ();
            let tsm_stride_0_tensor3 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor3 : isize = tsm_strides [1usize];
            let tsm_stride_2_tensor3 : isize = tsm_strides [2usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 2usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 , tsm_dim_1 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor3})});
            for i in 0usize .. tsm_dim_0 {
                for j in 0usize .. tsm_dim_1 {
                    let tsm_value = {
                        (* unsafe {& * tsm_ptr_tensor3 . offset ((:: tensorism :: Reindexing1 :: get_unchecked (& sup , :: tensorism :: Reindexing2 :: get_unchecked (& middle , :: tensorism :: Reindexing2 :: get_unchecked (& sub , i , j), j))as isize)* tsm_stride_0_tensor3 + (j as isize)* tsm_stride_1_tensor3 + (i as isize)* tsm_stride_2_tensor3)})
                    };
                    unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
                };
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );
}

#[test]
fn reverse_alias_format() {
    let format = format_new_ndarray!(for i let j = rev: i => tensor1[j, i]);

    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 1 ;
            let tsm_dim_1 = tsm_dim_0 ;
            if tsm_dim_1 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 0 {
                panic ! ("Dimensions are not matching between j = _ and tensor1[j, _]");
            };
            let tsm_ptr_tensor1 : * const _ = tensor1 . as_ptr ();
            let tsm_strides = tensor1 . strides ();
            let tsm_stride_0_tensor1 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor1 : isize = tsm_strides [1usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 1usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor1})});
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    let j = (tsm_dim_0 - 1 - i);
                    (* unsafe {& * tsm_ptr_tensor1 . offset ((j as isize)* tsm_stride_0_tensor1 + (i as isize)* tsm_stride_1_tensor1)})
                };
                unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );
}

#[test]
fn reverse_alias_generation() {
    let tensor1 = Array2::<i32>::from_shape_fn((5, 5), |(j, i)| (j as i32) * (i as i32 + 6));
    let expected = Array1::<i32>::from_vec(vec![24, 21, 16, 9, 0]);

    let result = new_ndarray!(for i let j = rev: i => tensor1[j, i]);
    assert_eq!(expected, result);

    let result = new_ndarray!(for i => tensor1[rev: i, i]);
    assert_eq!(expected, result);
}

#[test]
fn direct_alias_format() {
    let format = format_new_ndarray!(for i let j = i => tensor1[j, i]);

    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 1 ;
            let tsm_dim_1 = tsm_dim_0 ;
            if tsm_dim_1 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 0 {
                panic ! ("Dimensions are not matching between j = _ and tensor1[j, _]");
            };
            let tsm_ptr_tensor1 : * const _ = tensor1 . as_ptr ();
            let tsm_strides = tensor1 . strides ();
            let tsm_stride_0_tensor1 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor1 : isize = tsm_strides [1usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 1usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor1})});
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    let j = i ;
                    (* unsafe {& * tsm_ptr_tensor1 . offset ((j as isize)* tsm_stride_0_tensor1 + (i as isize)* tsm_stride_1_tensor1)})
                };
                unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );
}
