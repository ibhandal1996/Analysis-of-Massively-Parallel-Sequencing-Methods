??
??
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
?
AsString

input"T

output"
Ttype:
2	
"
	precisionint?????????"

scientificbool( "
shortestbool( "
widthint?????????"
fillstring 
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?"serve*2.8.0-dev202111102v1.12.1-67005-g70655959b1c8??

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
?
global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
o
input_example_tensorPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB 
?
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*'
valueBBA_1BC_1BG_1BT_1
j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 
?
ParseExample/ParseExampleV2ParseExampleV2input_example_tensor!ParseExample/ParseExampleV2/names'ParseExample/ParseExampleV2/sparse_keys&ParseExample/ParseExampleV2/dense_keys'ParseExample/ParseExampleV2/ragged_keysParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3*
Tdense
2*`
_output_shapesN
L:?????????:?????????:?????????:?????????**
dense_shapes
::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 
?
1linear/linear_model/A_1/weights/Initializer/zerosConst*2
_class(
&$loc:@linear/linear_model/A_1/weights*
_output_shapes

:*
dtype0*
valueB*    
?
linear/linear_model/A_1/weightsVarHandleOp*2
_class(
&$loc:@linear/linear_model/A_1/weights*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear/linear_model/A_1/weights
?
@linear/linear_model/A_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOplinear/linear_model/A_1/weights*
_output_shapes
: 
?
&linear/linear_model/A_1/weights/AssignAssignVariableOplinear/linear_model/A_1/weights1linear/linear_model/A_1/weights/Initializer/zeros*
dtype0
?
3linear/linear_model/A_1/weights/Read/ReadVariableOpReadVariableOplinear/linear_model/A_1/weights*
_output_shapes

:*
dtype0
?
1linear/linear_model/C_1/weights/Initializer/zerosConst*2
_class(
&$loc:@linear/linear_model/C_1/weights*
_output_shapes

:*
dtype0*
valueB*    
?
linear/linear_model/C_1/weightsVarHandleOp*2
_class(
&$loc:@linear/linear_model/C_1/weights*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear/linear_model/C_1/weights
?
@linear/linear_model/C_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOplinear/linear_model/C_1/weights*
_output_shapes
: 
?
&linear/linear_model/C_1/weights/AssignAssignVariableOplinear/linear_model/C_1/weights1linear/linear_model/C_1/weights/Initializer/zeros*
dtype0
?
3linear/linear_model/C_1/weights/Read/ReadVariableOpReadVariableOplinear/linear_model/C_1/weights*
_output_shapes

:*
dtype0
?
1linear/linear_model/G_1/weights/Initializer/zerosConst*2
_class(
&$loc:@linear/linear_model/G_1/weights*
_output_shapes

:*
dtype0*
valueB*    
?
linear/linear_model/G_1/weightsVarHandleOp*2
_class(
&$loc:@linear/linear_model/G_1/weights*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear/linear_model/G_1/weights
?
@linear/linear_model/G_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOplinear/linear_model/G_1/weights*
_output_shapes
: 
?
&linear/linear_model/G_1/weights/AssignAssignVariableOplinear/linear_model/G_1/weights1linear/linear_model/G_1/weights/Initializer/zeros*
dtype0
?
3linear/linear_model/G_1/weights/Read/ReadVariableOpReadVariableOplinear/linear_model/G_1/weights*
_output_shapes

:*
dtype0
?
1linear/linear_model/T_1/weights/Initializer/zerosConst*2
_class(
&$loc:@linear/linear_model/T_1/weights*
_output_shapes

:*
dtype0*
valueB*    
?
linear/linear_model/T_1/weightsVarHandleOp*2
_class(
&$loc:@linear/linear_model/T_1/weights*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear/linear_model/T_1/weights
?
@linear/linear_model/T_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOplinear/linear_model/T_1/weights*
_output_shapes
: 
?
&linear/linear_model/T_1/weights/AssignAssignVariableOplinear/linear_model/T_1/weights1linear/linear_model/T_1/weights/Initializer/zeros*
dtype0
?
3linear/linear_model/T_1/weights/Read/ReadVariableOpReadVariableOplinear/linear_model/T_1/weights*
_output_shapes

:*
dtype0
?
2linear/linear_model/bias_weights/Initializer/zerosConst*3
_class)
'%loc:@linear/linear_model/bias_weights*
_output_shapes
:*
dtype0*
valueB*    
?
 linear/linear_model/bias_weightsVarHandleOp*3
_class)
'%loc:@linear/linear_model/bias_weights*
_output_shapes
: *
dtype0*
shape:*1
shared_name" linear/linear_model/bias_weights
?
Alinear/linear_model/bias_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp linear/linear_model/bias_weights*
_output_shapes
: 
?
'linear/linear_model/bias_weights/AssignAssignVariableOp linear/linear_model/bias_weights2linear/linear_model/bias_weights/Initializer/zeros*
dtype0
?
4linear/linear_model/bias_weights/Read/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
?
Elinear/linear_model/linear/linear_model/linear/linear_model/A_1/ShapeShapeParseExample/ParseExampleV2*
T0*
_output_shapes
:
?
Slinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_sliceStridedSliceElinear/linear_model/linear/linear_model/linear/linear_model/A_1/ShapeSlinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_slice/stackUlinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_slice/stack_1Ulinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Olinear/linear_model/linear/linear_model/linear/linear_model/A_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/A_1/Reshape/shapePackMlinear/linear_model/linear/linear_model/linear/linear_model/A_1/strided_sliceOlinear/linear_model/linear/linear_model/linear/linear_model/A_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Glinear/linear_model/linear/linear_model/linear/linear_model/A_1/ReshapeReshapeParseExample/ParseExampleV2Mlinear/linear_model/linear/linear_model/linear/linear_model/A_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
[linear/linear_model/linear/linear_model/linear/linear_model/A_1/weighted_sum/ReadVariableOpReadVariableOplinear/linear_model/A_1/weights*
_output_shapes

:*
dtype0
?
Llinear/linear_model/linear/linear_model/linear/linear_model/A_1/weighted_sumMatMulGlinear/linear_model/linear/linear_model/linear/linear_model/A_1/Reshape[linear/linear_model/linear/linear_model/linear/linear_model/A_1/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:?????????
?
Elinear/linear_model/linear/linear_model/linear/linear_model/C_1/ShapeShapeParseExample/ParseExampleV2:1*
T0*
_output_shapes
:
?
Slinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_sliceStridedSliceElinear/linear_model/linear/linear_model/linear/linear_model/C_1/ShapeSlinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_slice/stackUlinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_slice/stack_1Ulinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Olinear/linear_model/linear/linear_model/linear/linear_model/C_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/C_1/Reshape/shapePackMlinear/linear_model/linear/linear_model/linear/linear_model/C_1/strided_sliceOlinear/linear_model/linear/linear_model/linear/linear_model/C_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Glinear/linear_model/linear/linear_model/linear/linear_model/C_1/ReshapeReshapeParseExample/ParseExampleV2:1Mlinear/linear_model/linear/linear_model/linear/linear_model/C_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
[linear/linear_model/linear/linear_model/linear/linear_model/C_1/weighted_sum/ReadVariableOpReadVariableOplinear/linear_model/C_1/weights*
_output_shapes

:*
dtype0
?
Llinear/linear_model/linear/linear_model/linear/linear_model/C_1/weighted_sumMatMulGlinear/linear_model/linear/linear_model/linear/linear_model/C_1/Reshape[linear/linear_model/linear/linear_model/linear/linear_model/C_1/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:?????????
?
Elinear/linear_model/linear/linear_model/linear/linear_model/G_1/ShapeShapeParseExample/ParseExampleV2:2*
T0*
_output_shapes
:
?
Slinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_sliceStridedSliceElinear/linear_model/linear/linear_model/linear/linear_model/G_1/ShapeSlinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_slice/stackUlinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_slice/stack_1Ulinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Olinear/linear_model/linear/linear_model/linear/linear_model/G_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/G_1/Reshape/shapePackMlinear/linear_model/linear/linear_model/linear/linear_model/G_1/strided_sliceOlinear/linear_model/linear/linear_model/linear/linear_model/G_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Glinear/linear_model/linear/linear_model/linear/linear_model/G_1/ReshapeReshapeParseExample/ParseExampleV2:2Mlinear/linear_model/linear/linear_model/linear/linear_model/G_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
[linear/linear_model/linear/linear_model/linear/linear_model/G_1/weighted_sum/ReadVariableOpReadVariableOplinear/linear_model/G_1/weights*
_output_shapes

:*
dtype0
?
Llinear/linear_model/linear/linear_model/linear/linear_model/G_1/weighted_sumMatMulGlinear/linear_model/linear/linear_model/linear/linear_model/G_1/Reshape[linear/linear_model/linear/linear_model/linear/linear_model/G_1/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:?????????
?
Elinear/linear_model/linear/linear_model/linear/linear_model/T_1/ShapeShapeParseExample/ParseExampleV2:3*
T0*
_output_shapes
:
?
Slinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_sliceStridedSliceElinear/linear_model/linear/linear_model/linear/linear_model/T_1/ShapeSlinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_slice/stackUlinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_slice/stack_1Ulinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Olinear/linear_model/linear/linear_model/linear/linear_model/T_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Mlinear/linear_model/linear/linear_model/linear/linear_model/T_1/Reshape/shapePackMlinear/linear_model/linear/linear_model/linear/linear_model/T_1/strided_sliceOlinear/linear_model/linear/linear_model/linear/linear_model/T_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Glinear/linear_model/linear/linear_model/linear/linear_model/T_1/ReshapeReshapeParseExample/ParseExampleV2:3Mlinear/linear_model/linear/linear_model/linear/linear_model/T_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
[linear/linear_model/linear/linear_model/linear/linear_model/T_1/weighted_sum/ReadVariableOpReadVariableOplinear/linear_model/T_1/weights*
_output_shapes

:*
dtype0
?
Llinear/linear_model/linear/linear_model/linear/linear_model/T_1/weighted_sumMatMulGlinear/linear_model/linear/linear_model/linear/linear_model/T_1/Reshape[linear/linear_model/linear/linear_model/linear/linear_model/T_1/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasAddNLlinear/linear_model/linear/linear_model/linear/linear_model/A_1/weighted_sumLlinear/linear_model/linear/linear_model/linear/linear_model/C_1/weighted_sumLlinear/linear_model/linear/linear_model/linear/linear_model/G_1/weighted_sumLlinear/linear_model/linear/linear_model/linear/linear_model/T_1/weighted_sum*
N*
T0*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
?
Hlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sumBiasAddPlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasWlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:?????????
M
bias/tagConst*
_output_shapes
: *
dtype0*
valueB
 Bbias
p
bias/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
O
biasHistogramSummarybias/tagbias/ReadVariableOp*
_output_shapes
: 
?
,zero_fraction/total_size/Size/ReadVariableOpReadVariableOplinear/linear_model/A_1/weights*
_output_shapes

:*
dtype0
_
zero_fraction/total_size/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
.zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOplinear/linear_model/C_1/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
.zero_fraction/total_size/Size_2/ReadVariableOpReadVariableOplinear/linear_model/G_1/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_2Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
.zero_fraction/total_size/Size_3/ReadVariableOpReadVariableOplinear/linear_model/T_1/weights*
_output_shapes

:*
dtype0
a
zero_fraction/total_size/Size_3Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
zero_fraction/total_size/AddNAddNzero_fraction/total_size/Sizezero_fraction/total_size/Size_1zero_fraction/total_size/Size_2zero_fraction/total_size/Size_3*
N*
T0	*
_output_shapes
: 
`
zero_fraction/total_zero/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
zero_fraction/total_zero/EqualEqualzero_fraction/total_size/Sizezero_fraction/total_zero/Const*
T0	*
_output_shapes
: 
?
#zero_fraction/total_zero/zero_countIfzero_fraction/total_zero/Equallinear/linear_model/A_1/weightszero_fraction/total_size/Size*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*A
else_branch2R0
.zero_fraction_total_zero_zero_count_false_1659*
output_shapes
: *@
then_branch1R/
-zero_fraction_total_zero_zero_count_true_1658
~
,zero_fraction/total_zero/zero_count/IdentityIdentity#zero_fraction/total_zero/zero_count*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
 zero_fraction/total_zero/Equal_1Equalzero_fraction/total_size/Size_1 zero_fraction/total_zero/Const_1*
T0	*
_output_shapes
: 
?
%zero_fraction/total_zero/zero_count_1If zero_fraction/total_zero/Equal_1linear/linear_model/C_1/weightszero_fraction/total_size/Size_1*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_1_false_1702*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_1_true_1701
?
.zero_fraction/total_zero/zero_count_1/IdentityIdentity%zero_fraction/total_zero/zero_count_1*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
 zero_fraction/total_zero/Equal_2Equalzero_fraction/total_size/Size_2 zero_fraction/total_zero/Const_2*
T0	*
_output_shapes
: 
?
%zero_fraction/total_zero/zero_count_2If zero_fraction/total_zero/Equal_2linear/linear_model/G_1/weightszero_fraction/total_size/Size_2*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_2_false_1745*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_2_true_1744
?
.zero_fraction/total_zero/zero_count_2/IdentityIdentity%zero_fraction/total_zero/zero_count_2*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
 zero_fraction/total_zero/Equal_3Equalzero_fraction/total_size/Size_3 zero_fraction/total_zero/Const_3*
T0	*
_output_shapes
: 
?
%zero_fraction/total_zero/zero_count_3If zero_fraction/total_zero/Equal_3linear/linear_model/T_1/weightszero_fraction/total_size/Size_3*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*C
else_branch4R2
0zero_fraction_total_zero_zero_count_3_false_1788*
output_shapes
: *B
then_branch3R1
/zero_fraction_total_zero_zero_count_3_true_1787
?
.zero_fraction/total_zero/zero_count_3/IdentityIdentity%zero_fraction/total_zero/zero_count_3*
T0*
_output_shapes
: 
?
zero_fraction/total_zero/AddNAddN,zero_fraction/total_zero/zero_count/Identity.zero_fraction/total_zero/zero_count_1/Identity.zero_fraction/total_zero/zero_count_2/Identity.zero_fraction/total_zero/zero_count_3/Identity*
N*
T0*
_output_shapes
: 
y
"zero_fraction/compute/float32_sizeCastzero_fraction/total_size/AddN*

DstT0*

SrcT0	*
_output_shapes
: 
?
zero_fraction/compute/truedivRealDivzero_fraction/total_zero/AddN"zero_fraction/compute/float32_size*
T0*
_output_shapes
: 
n
"zero_fraction/zero_fraction_or_nanIdentityzero_fraction/compute/truediv*
T0*
_output_shapes
: 
v
fraction_of_zero_weights/tagsConst*
_output_shapes
: *
dtype0*)
value B Bfraction_of_zero_weights
?
fraction_of_zero_weightsScalarSummaryfraction_of_zero_weights/tags"zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 
?
head/logits/ShapeShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
?
head/predictions/probabilitiesSoftmaxHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*'
_output_shapes
:?????????
o
$head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/class_idsArgMaxHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:?????????
j
head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:?????????
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:?????????
?
head/predictions/ShapeShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
p
&head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
p
&head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
^
head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
c
!head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*
T0*'
_output_shapes
:?????????
?
head/predictions/Shape_1ShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
head/predictions/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
head/predictions/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
`
head/predictions/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_2
ExpandDimshead/predictions/range_1!head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
m
head/predictions/AsStringAsStringhead/predictions/ExpandDims_2*
T0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/Tile_1Tilehead/predictions/AsString!head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:?????????
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
d
head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
d
head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
R
head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
R
head/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
R
head/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
U
head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
g
head/ExpandDims
ExpandDims
head/rangehead/ExpandDims/dim*
T0*
_output_shapes

:
S
head/AsStringAsStringhead/ExpandDims*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
N*
T0*
_output_shapes
:
g
	head/TileTilehead/AsStringhead/Tile/multiples*
T0*'
_output_shapes
:?????????

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp\part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Bglobal_stepBlinear/linear_model/A_1/weightsBlinear/linear_model/C_1/weightsBlinear/linear_model/G_1/weightsBlinear/linear_model/T_1/weightsB linear/linear_model/bias_weights
~
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step/Read/ReadVariableOp3linear/linear_model/A_1/weights/Read/ReadVariableOp3linear/linear_model/C_1/weights/Read/ReadVariableOp3linear/linear_model/G_1/weights/Read/ReadVariableOp3linear/linear_model/T_1/weights/Read/ReadVariableOp4linear/linear_model/bias_weights/Read/ReadVariableOp"/device:CPU:0*
dtypes

2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Bglobal_stepBlinear/linear_model/A_1/weightsBlinear/linear_model/C_1/weightsBlinear/linear_model/G_1/weightsBlinear/linear_model/T_1/weightsB linear/linear_model/bias_weights
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	
N
save/Identity_1Identitysave/RestoreV2*
T0	*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpglobal_stepsave/Identity_1*
dtype0	
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
j
save/AssignVariableOp_1AssignVariableOplinear/linear_model/A_1/weightssave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
j
save/AssignVariableOp_2AssignVariableOplinear/linear_model/C_1/weightssave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
j
save/AssignVariableOp_3AssignVariableOplinear/linear_model/G_1/weightssave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
j
save/AssignVariableOp_4AssignVariableOplinear/linear_model/T_1/weightssave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
k
save/AssignVariableOp_5AssignVariableOp linear/linear_model/bias_weightssave/Identity_6*
dtype0
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5
-
save/restore_allNoOp^save/restore_shard?~
?
?
0zero_fraction_total_zero_zero_count_3_false_1788N
<zero_fraction_readvariableop_linear_linear_model_t_1_weights:(
$cast_zero_fraction_total_size_size_3	
mul??
zero_fraction/ReadVariableOpReadVariableOp<zero_fraction_readvariableop_linear_linear_model_t_1_weights*
_output_shapes

:*
dtype0T
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R_
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R?????
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: ?
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_1798*
output_shapes
: */
then_branch R
zero_fraction_cond_true_1797e
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: ?
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: ?
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: |
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: q
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: b
CastCast$cast_zero_fraction_total_size_size_3*

DstT0*

SrcT0	*
_output_shapes
: @
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T0"
mul	mul_0:z:0*
_input_shapes
: : :

_output_shapes
: 
?
?
0zero_fraction_total_zero_zero_count_2_false_1745N
<zero_fraction_readvariableop_linear_linear_model_g_1_weights:(
$cast_zero_fraction_total_size_size_2	
mul??
zero_fraction/ReadVariableOpReadVariableOp<zero_fraction_readvariableop_linear_linear_model_g_1_weights*
_output_shapes

:*
dtype0T
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R_
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R?????
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: ?
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_1755*
output_shapes
: */
then_branch R
zero_fraction_cond_true_1754e
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: ?
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: ?
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: |
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: q
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: b
CastCast$cast_zero_fraction_total_size_size_2*

DstT0*

SrcT0	*
_output_shapes
: @
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T0"
mul	mul_0:z:0*
_input_shapes
: : :

_output_shapes
: 
?
`
/zero_fraction_total_zero_zero_count_1_true_1701
placeholder
placeholder_1		
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    "
constConst:output:0*
_input_shapes
: : :

_output_shapes
: 
?
y
zero_fraction_cond_false_16697
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: "C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
?
a
zero_fraction_cond_true_17977
3count_nonzero_notequal_zero_fraction_readvariableop
cast	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: b
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: "
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
?
`
/zero_fraction_total_zero_zero_count_2_true_1744
placeholder
placeholder_1		
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    "
constConst:output:0*
_input_shapes
: : :

_output_shapes
: 
?
?
0zero_fraction_total_zero_zero_count_1_false_1702N
<zero_fraction_readvariableop_linear_linear_model_c_1_weights:(
$cast_zero_fraction_total_size_size_1	
mul??
zero_fraction/ReadVariableOpReadVariableOp<zero_fraction_readvariableop_linear_linear_model_c_1_weights*
_output_shapes

:*
dtype0T
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R_
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R?????
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: ?
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_1712*
output_shapes
: */
then_branch R
zero_fraction_cond_true_1711e
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: ?
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: ?
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: |
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: q
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: b
CastCast$cast_zero_fraction_total_size_size_1*

DstT0*

SrcT0	*
_output_shapes
: @
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T0"
mul	mul_0:z:0*
_input_shapes
: : :

_output_shapes
: 
?
y
zero_fraction_cond_false_17987
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: "C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
?
^
-zero_fraction_total_zero_zero_count_true_1658
placeholder
placeholder_1		
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    "
constConst:output:0*
_input_shapes
: : :

_output_shapes
: 
?
a
zero_fraction_cond_true_17117
3count_nonzero_notequal_zero_fraction_readvariableop
cast	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: b
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: "
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
?
?
.zero_fraction_total_zero_zero_count_false_1659N
<zero_fraction_readvariableop_linear_linear_model_a_1_weights:&
"cast_zero_fraction_total_size_size	
mul??
zero_fraction/ReadVariableOpReadVariableOp<zero_fraction_readvariableop_linear_linear_model_a_1_weights*
_output_shapes

:*
dtype0T
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R_
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R?????
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: ?
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_1669*
output_shapes
: */
then_branch R
zero_fraction_cond_true_1668e
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: ?
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: ?
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: |
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: q
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: `
CastCast"cast_zero_fraction_total_size_size*

DstT0*

SrcT0	*
_output_shapes
: @
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T0"
mul	mul_0:z:0*
_input_shapes
: : :

_output_shapes
: 
?
a
zero_fraction_cond_true_17547
3count_nonzero_notequal_zero_fraction_readvariableop
cast	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: b
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: "
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
?
`
/zero_fraction_total_zero_zero_count_3_true_1787
placeholder
placeholder_1		
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    "
constConst:output:0*
_input_shapes
: : :

_output_shapes
: 
?
y
zero_fraction_cond_false_17557
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: "C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
?
y
zero_fraction_cond_false_17127
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: "C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
?
a
zero_fraction_cond_true_16687
3count_nonzero_notequal_zero_fraction_readvariableop
cast	X
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: b
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: "
castCast:y:0*
_input_shapes

::$  

_output_shapes

:"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"3
	summaries&
$
bias:0
fraction_of_zero_weights:0"?
trainable_variables??
?
!linear/linear_model/A_1/weights:0&linear/linear_model/A_1/weights/Assign5linear/linear_model/A_1/weights/Read/ReadVariableOp:0(23linear/linear_model/A_1/weights/Initializer/zeros:08
?
!linear/linear_model/C_1/weights:0&linear/linear_model/C_1/weights/Assign5linear/linear_model/C_1/weights/Read/ReadVariableOp:0(23linear/linear_model/C_1/weights/Initializer/zeros:08
?
!linear/linear_model/G_1/weights:0&linear/linear_model/G_1/weights/Assign5linear/linear_model/G_1/weights/Read/ReadVariableOp:0(23linear/linear_model/G_1/weights/Initializer/zeros:08
?
!linear/linear_model/T_1/weights:0&linear/linear_model/T_1/weights/Assign5linear/linear_model/T_1/weights/Read/ReadVariableOp:0(23linear/linear_model/T_1/weights/Initializer/zeros:08
?
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08"?
	variables??
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
?
!linear/linear_model/A_1/weights:0&linear/linear_model/A_1/weights/Assign5linear/linear_model/A_1/weights/Read/ReadVariableOp:0(23linear/linear_model/A_1/weights/Initializer/zeros:08
?
!linear/linear_model/C_1/weights:0&linear/linear_model/C_1/weights/Assign5linear/linear_model/C_1/weights/Read/ReadVariableOp:0(23linear/linear_model/C_1/weights/Initializer/zeros:08
?
!linear/linear_model/G_1/weights:0&linear/linear_model/G_1/weights/Assign5linear/linear_model/G_1/weights/Read/ReadVariableOp:0(23linear/linear_model/G_1/weights/Initializer/zeros:08
?
!linear/linear_model/T_1/weights:0&linear/linear_model/T_1/weights/Assign5linear/linear_model/T_1/weights/Read/ReadVariableOp:0(23linear/linear_model/T_1/weights/Initializer/zeros:08
?
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08*?
classification?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify*?
predict?
5
examples)
input_example_tensor:0??????????
all_class_ids.
head/predictions/Tile:0??????????
all_classes0
head/predictions/Tile_1:0?????????A
	class_ids4
head/predictions/ExpandDims:0	?????????@
classes5
head/predictions/str_classes:0?????????k
logitsa
Jlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum:0?????????H
probabilities7
 head/predictions/probabilities:0?????????tensorflow/serving/predict*?
serving_default?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify