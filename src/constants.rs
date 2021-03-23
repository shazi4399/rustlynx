pub const SIZEOF_MEGABYTE: usize = 1 << 20;
pub const SIZEOF_U128: usize = 1 << 4;
pub const SIZEOF_U64: usize = 1 << 3;
pub const SIZEOF_U32: usize = 1 << 2;
pub const SIZEOF_U16: usize = 1 << 1;
pub const SIZEOF_U8: usize = 1;
pub const BITLENGTH: usize = 64;

pub const TEST_CR_WRAPPING_SEL_VEC_U64_0: (u64, u64) = (
	0x91726f49cade6a19,
	0x1726F49CADE6A1AC,
);

pub const TEST_CR_WRAPPING_SEL_VEC_U64_1: (u64, u64) = (
	0x91726f49cade6a1a,
	0x1726F49CADE6A1AC,
);

pub const TEST_CR_WRAPPING_U64_0: (u64, u64, u64) = (
		0x91726f49cade6a1a,
		0xe5b959fd785813fe,
		0xffdbcf4a9eb6e119,
	);

pub const TEST_CR_WRAPPING_U64_1: (u64, u64, u64) = (
		0xa57a5db4030e7993,
		0x7c5d53166af86ce0,
		0x4e351284f39d0eed,
	);

pub const TEST_CR_XOR_U128_0: (u128, u128, u128) = (
		0x602113ab721dfbb0a41aec62e0aa9d16,
		0x3347a4e096013fd85938a2f1cdf35047,
		0x4e8d7fb29621dcbb373d03cd71d77483,
	);

pub const TEST_CR_XOR_U128_1: (u128, u128, u128) = (
		0xb5f0592ca4dc5b3e3628159382e3a1c7,
		0x5d63a3d79b3dd43c48e20ce7a80e14ef,
		0x0a8d7db592217c3f272fabdd119e7003,
	);

pub const TEST_CR_XOR_U128_PAIRWISE_0: (u128, u128) = (
		0xa56d161e9dbb7fed1c581d5357a45b00,
		0x8a802aa28820a2aa2280a02a0a00a2a0,
	);


pub const TEST_CR_XOR_U128_PAIRWISE_1: (u128, u128) = (
		0x614f0f39a4f1896bf0a119119fea385a,
		0x0a802aa0a82002aaaa20a02a8a08a0a0,
	);



