// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

static BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

static BIT_TABLE: [u8; 256] = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
];

/// Returns the nearest number that is `>=` than `num` and is a multiple of 64.
#[inline]
pub fn round_upto_multiple_of_64(num: i64) -> i64 {
    let force_carry_addend = 63;
    let truncate_bitmask = !63;
    let max_roundable_num = ::std::i64::MAX - 64;
    if num < max_roundable_num {
        return (num + force_carry_addend) & truncate_bitmask;
    }
    num
}

/// Returns whether bit at position `i` in `data` is set or not.
#[inline]
pub fn get_bit(data: &[u8], i: usize) -> bool {
    (data[i / 8] & BIT_MASK[i % 8]) != 0
}

/// Sets bit at position `i` for `data`.
#[inline]
pub fn set_bit(data: &mut [u8], i: usize) {
    data[i / 8] |= BIT_MASK[i % 8]
}

/// Returns the number of 1-bits in `data`.
#[inline]
pub fn count_set_bits(data: &[u8]) -> i64 {
    let mut count: i64 = 0;
    for u in data {
        count += BIT_TABLE[*u as usize] as i64;
    }
    count
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_round_upto_multiple_of_64() {
        assert_eq!(0, round_upto_multiple_of_64(0));
        assert_eq!(64, round_upto_multiple_of_64(1));
        assert_eq!(64, round_upto_multiple_of_64(63));
        assert_eq!(64, round_upto_multiple_of_64(64));
        assert_eq!(128, round_upto_multiple_of_64(65));
        assert_eq!(
            ::std::i64::MAX - 64,
            round_upto_multiple_of_64(::std::i64::MAX - 64)
        );
    }

    #[test]
    fn test_get_bit() {
        // 00001101
        assert_eq!(true, get_bit(&[0b00001101], 0));
        assert_eq!(false, get_bit(&[0b00001101], 1));
        assert_eq!(true, get_bit(&[0b00001101], 2));
        assert_eq!(true, get_bit(&[0b00001101], 3));

        // 01001001 01010010
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 0));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 1));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 2));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 3));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 4));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 5));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 6));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 7));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 8));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 9));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 10));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 11));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 12));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 13));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 14));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 15));
    }

    #[test]
    fn test_set_bit() {
        let mut b = [0b00000000];
        set_bit(&mut b, 0);
        assert_eq!([0b00000001], b);
        set_bit(&mut b, 2);
        assert_eq!([0b00000101], b);
        set_bit(&mut b, 5);
        assert_eq!([0b00100101], b);
    }

    #[test]
    fn test_get_set_bit_roundtrip() {
        const NUM_BYTES: usize = 10;
        const NUM_SETS: usize = 10;

        let mut buffer: Vec<u8> = Vec::with_capacity(NUM_BYTES);
        unsafe {
            buffer.set_len(NUM_BYTES);
        }
        let mut v = HashSet::new();
        let mut rng = thread_rng();
        for _ in 0..NUM_SETS {
            let offset = rng.gen_range(0, 8 * NUM_BYTES);
            v.insert(offset);
            set_bit(&mut buffer[..], offset);
        }

        for i in 0..NUM_BYTES * 8 {
            assert_eq!(v.contains(&i), get_bit(&buffer[..], i));
        }
    }

    #[test]
    fn test_count_bits_slice() {
        assert_eq!(0, count_set_bits(&[0b00000000]));
        assert_eq!(8, count_set_bits(&[0b11111111]));
        assert_eq!(3, count_set_bits(&[0b00001101]));
        assert_eq!(6, count_set_bits(&[0b01001001, 0b01010010]));
    }
}
