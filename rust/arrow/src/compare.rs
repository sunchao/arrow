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

use crate::array::*;
use crate::array_data::*;
use crate::buffer::Buffer;
use crate::datatypes::*;
use crate::memory::memcmp;
use crate::util::bit_util;

/// Trait for `Array` equality.
pub trait ArrayEqual {
    /// Returns true if this array is equal to the `other` array
    fn equals(&self, other: &dyn Array) -> bool;

    /// Returns true if the range [start_idx, end_idx) is equal to
    /// [other_start_idx, other_start_idx + end_idx - start_idx) in the `other` array
    fn range_equals(
        &self,
        other: &ArrayRef,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool;
}

trait PrimitiveArrayTrait<T: ArrowPrimitiveType> {
    fn values(&self) -> Buffer;
    fn value(&self, i: usize) -> T::Native;
}

impl<T: ArrowPrimitiveType> PrimitiveArrayTrait<T> for PrimitiveArray<T> {
    default fn values(&self) -> Buffer {
        unimplemented!()
    }

    default fn value(&self, _: usize) -> T::Native {
        unimplemented!()
    }
}

impl<T: ArrowNumericType> PrimitiveArrayTrait<T> for PrimitiveArray<T> {
    fn values(&self) -> Buffer {
        self.values()
    }

    fn value(&self, i: usize) -> T::Native {
        self.value(i)
    }
}

impl PrimitiveArrayTrait<BooleanType> for BooleanArray {
    fn values(&self) -> Buffer {
        self.values()
    }

    fn value(&self, i: usize) -> bool {
        self.value(i)
    }
}

impl<T: ArrowPrimitiveType> ArrayEqual for PrimitiveArray<T> {
    default fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let value_buf = self.data_ref().buffers()[0].clone();
        let other_value_buf = other.data_ref().buffers()[0].clone();
        let byte_width = T::get_bit_width() / 8;

        if self.null_count() > 0 {
            let values = value_buf.data();
            let other_values = other_value_buf.data();

            for i in 0..self.len() {
                if self.is_valid(i) {
                    let start = (i + self.offset()) * byte_width;
                    let data = &values[start..(start + byte_width)];
                    let other_start = (i + other.offset()) * byte_width;
                    let other_data =
                        &other_values[other_start..(other_start + byte_width)];
                    if data != other_data {
                        return false;
                    }
                }
            }
        } else {
            let start = self.offset() * byte_width;
            let other_start = other.offset() * byte_width;
            let len = self.len() * byte_width;
            let data = &value_buf.data()[start..(start + len)];
            let other_data = &other_value_buf.data()[other_start..(other_start + len)];
            if data != other_data {
                return false;
            }
        }

        true
    }

    default fn range_equals(
        &self,
        other: &ArrayRef,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        let other = other.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(j);
            if is_null != other_is_null || (!is_null && self.value(i) != other.value(j)) {
                return false;
            }
            j += 1;
        }

        true
    }
}

impl ArrayEqual for BooleanArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let values = self.data_ref().buffers()[0].data();
        let other_values = other.data_ref().buffers()[0].data();

        // TODO: we can do this more efficiently if all values are not-null
        for i in 0..self.len() {
            if self.is_valid(i) {
                if bit_util::get_bit(values, i + self.offset())
                    != bit_util::get_bit(other_values, i + other.offset())
                {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: ArrowNumericType> PartialEq for PrimitiveArray<T> {
    fn eq(&self, other: &PrimitiveArray<T>) -> bool {
        self.equals(other)
    }
}

impl ArrayEqual for ListArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<ListArray>().unwrap();

        // Check if offsets differ
        if self.offset() == 0 && other.offset() == 0 {
            let offset_data = &self.data_ref().buffers()[0];
            let other_offset_data = &other.data_ref().buffers()[0];
            return offset_data.data()[0..(self.len() * 4)]
                == other_offset_data.data()[0..(other.len() * 4)];
        }

        // The expensive case
        for i in 0..self.len() + 1 {
            if self.value_offset_at(i) - self.value_offset_at(0)
                != other.value_offset_at(i) - other.value_offset_at(0)
            {
                return false;
            }
        }

        let array = self.values();

        if !array.range_equals(
            &other.values(),
            self.value_offset(0) as usize,
            self.value_offset(self.len()) as usize,
            other.value_offset(0) as usize,
        ) {
            return false;
        }

        true
    }

    fn range_equals(
        &self,
        other: &ArrayRef,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        let other = other.as_any().downcast_ref::<ListArray>().unwrap();
        let values = self.values();
        let other_values = other.values();

        let mut o_i = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(o_i);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }

            let start_offset = self.value_offset(i) as usize;
            let end_offset = self.value_offset(i + 1) as usize;
            let other_start_offset = other.value_offset(o_i) as usize;
            let other_end_offset = other.value_offset(o_i + 1) as usize;

            if end_offset - start_offset != other_end_offset - other_start_offset {
                return false;
            }

            if !values.range_equals(
                &other_values,
                start_offset,
                end_offset,
                other_start_offset,
            ) {
                return false;
            }

            o_i += 1;
        }

        true
    }
}

impl ArrayEqual for BinaryArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<BinaryArray>().unwrap();

        // Check if offsets differ
        if self.offset() == 0 && other.offset() == 0 {
            let offset_data = &self.data_ref().buffers()[0];
            let other_offset_data = &other.data_ref().buffers()[0];
            return offset_data.data()[0..(self.len() * 4)]
                == other_offset_data.data()[0..(other.len() * 4)];
        }

        // The expensive case
        for i in 0..self.len() + 1 {
            if self.value_offset_at(i) - self.value_offset_at(0)
                != other.value_offset_at(i) - other.value_offset_at(0)
            {
                return false;
            }
        }

        // TODO: handle null & length == 0 case?

        let value_data = self.value_data.get();
        let other_value_data = other.value_data.get();

        if self.null_count() == 0 {
            // No offset in both - just do memcmp
            if self.offset() == 0 && other.offset() == 0 {
                unsafe {
                    return memcmp(
                        value_data,
                        other_value_data,
                        self.value_offset(self.len()) as usize,
                    ) == 0;
                }
            } else {
                let total_bytes = self.value_offset(self.len()) - self.value_offset(0);
                unsafe {
                    // TODO: double check this and test.
                    return memcmp(
                        value_data.offset(self.value_offset(0) as isize),
                        other_value_data.offset(other.value_offset(0) as isize),
                        total_bytes as usize,
                    ) == 0;
                }
            }
        } else {
            for i in 0..self.len() {
                if self.is_null(i) {
                    continue;
                }
                unsafe {
                    if memcmp(
                        value_data.offset(self.value_offset(i) as isize),
                        other_value_data.offset(other.value_offset(i) as isize),
                        self.value_length(i) as usize,
                    ) != 0
                    {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn range_equals(
        &self,
        other: &ArrayRef,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        let other = other.as_any().downcast_ref::<BinaryArray>().unwrap();

        let mut o_i = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(o_i);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }

            let start_offset = self.value_offset(i) as usize;
            let end_offset = self.value_offset(i + 1) as usize;
            let other_start_offset = other.value_offset(o_i) as usize;
            let other_end_offset = other.value_offset(o_i + 1) as usize;

            if end_offset - start_offset != other_end_offset - other_start_offset {
                return false;
            }

            let value_data = self.value_data.get();
            let other_value_data = other.value_data.get();

            if end_offset - start_offset > 0 {
                unsafe {
                    if memcmp(
                        value_data.offset(start_offset as isize),
                        other_value_data.offset(other_start_offset as isize),
                        end_offset - start_offset,
                    ) != 0
                    {
                        return false;
                    }
                }
            }

            o_i += 1;
        }

        true
    }
}

impl ArrayEqual for StructArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<StructArray>().unwrap();

        for i in 0..self.len() {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(i);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }
            for j in 0..self.num_columns() {
                if !self.column(j).range_equals(other.column(j), i, i + 1, i) {
                    return false;
                }
            }
        }

        true
    }

    fn range_equals(
        &self,
        other: &ArrayRef,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        let other = other.as_any().downcast_ref::<StructArray>().unwrap();

        let mut o_i = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(i);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }
            for j in 0..self.num_columns() {
                if !self.column(j).range_equals(other.column(j), i, i + 1, o_i) {
                    return false;
                }
            }

            o_i += 1;
        }

        true
    }
}

// Comparing the common basic fields between the two arrays.
fn base_equal(this: &ArrayDataRef, other: &ArrayDataRef) -> bool {
    if this.data_type() != other.data_type() {
        return false;
    }
    if this.len != other.len {
        return false;
    }
    if this.null_count != other.null_count {
        return false;
    }
    if this.null_count > 0 {
        let null_bitmap = this.null_bitmap().as_ref().unwrap();
        let other_null_bitmap = other.null_bitmap().as_ref().unwrap();
        let null_buf = null_bitmap.bits.data();
        let other_null_buf = other_null_bitmap.bits.data();
        for i in 0..this.len() {
            if bit_util::get_bit(null_buf, i + this.offset())
                != bit_util::get_bit(other_null_buf, i + other.offset())
            {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::builder::{
        ArrayBuilder, BinaryBuilder, Int32Builder, ListBuilder, StructBuilder,
    };

    #[test]
    fn test_primitive_equal() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let b = Int32Array::from(vec![1, 2, 3]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = Int32Array::from(vec![1, 2, 4]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where null_count > 0

        let a = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        let b = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = Int32Array::from(vec![Some(1), None, None, Some(3)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let b = Int32Array::from(vec![Some(1), None, Some(2), Some(4)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a_slice = a.slice(1, 2);
        let b_slice = b.slice(1, 2);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));
    }

    #[test]
    fn test_boolean_equal() {
        let a = BooleanArray::from(vec![false, false, true]);
        let b = BooleanArray::from(vec![false, false, true]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = BooleanArray::from(vec![false, false, false]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where null_count > 0

        let a = BooleanArray::from(vec![Some(false), None, None, Some(true)]);
        let b = BooleanArray::from(vec![Some(false), None, None, Some(true)]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = BooleanArray::from(vec![None, None, None, Some(true)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let b = BooleanArray::from(vec![Some(true), None, None, Some(true)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a = BooleanArray::from(vec![false, true, false, true, false, false, true]);
        let b = BooleanArray::from(vec![false, false, false, true, false, true, true]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let a_slice = a.slice(2, 3);
        let b_slice = b.slice(2, 3);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));

        let a_slice = a.slice(3, 4);
        let b_slice = b.slice(3, 4);
        assert!(!a_slice.equals(&*b_slice));
        assert!(!b_slice.equals(&*a_slice));
    }

    #[test]
    fn test_list_equal() {
        let mut a_builder = ListBuilder::new(Int32Builder::new(10));
        let mut b_builder = ListBuilder::new(Int32Builder::new(10));

        a_builder.values().append_slice(&[1, 2, 3]).expect("");
        a_builder.append(true).expect("");
        a_builder.values().append_slice(&[4, 5]).expect("");
        a_builder.append(true).expect("");

        b_builder.values().append_slice(&[1, 2, 3]).expect("");
        b_builder.append(true).expect("");
        b_builder.values().append_slice(&[4, 5]).expect("");
        b_builder.append(true).expect("");

        let a = a_builder.finish();
        let b = b_builder.finish();
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        b_builder.values().append_slice(&[1, 2, 3]).expect("");
        a_builder.append(true).expect("");
        b_builder.values().append_slice(&[4, 5, 6]).expect("");
        a_builder.append(true).expect("");
        let b = b_builder.finish();

        assert!(!a.equals(&b));
        assert!(!b.equals(&a));
    }

    #[test]
    fn test_struct_equal() {
        let string_builder = BinaryBuilder::new(5);
        let int_builder = Int32Builder::new(5);

        let mut fields = Vec::new();
        let mut field_builders = Vec::new();
        fields.push(Field::new("f1", DataType::Utf8, false));
        field_builders.push(Box::new(string_builder) as Box<ArrayBuilder>);
        fields.push(Field::new("f2", DataType::Int32, false));
        field_builders.push(Box::new(int_builder) as Box<ArrayBuilder>);

        let mut builder = StructBuilder::new(fields, field_builders);

        let a = {
            let string_builder = builder.field_builder::<BinaryBuilder>(0).expect("");
            string_builder.append_string("joe").unwrap();
            string_builder.append_null().unwrap();
            string_builder.append_null().unwrap();
            string_builder.append_string("mark").unwrap();
            string_builder.append_string("doe").unwrap();

            let int_builder = builder.field_builder::<Int32Builder>(1).expect("");
            int_builder.append_value(1).unwrap();
            int_builder.append_value(2).unwrap();
            int_builder.append_null().unwrap();
            int_builder.append_value(4).unwrap();
            int_builder.append_value(5).unwrap();

            builder.append(true).unwrap();
            builder.append(true).unwrap();
            builder.append_null().unwrap();
            builder.append(true).unwrap();
            builder.append(true).unwrap();

            builder.finish()
        };

        let b = {
            let string_builder = builder.field_builder::<BinaryBuilder>(0).expect("");
            string_builder.append_string("joe").unwrap();
            string_builder.append_null().unwrap();
            string_builder.append_null().unwrap();
            string_builder.append_string("mark").unwrap();
            string_builder.append_string("doe").unwrap();

            let int_builder = builder.field_builder::<Int32Builder>(1).expect("");
            int_builder.append_value(1).unwrap();
            int_builder.append_value(2).unwrap();
            int_builder.append_null().unwrap();
            int_builder.append_value(4).unwrap();
            int_builder.append_value(5).unwrap();

            builder.append(true).unwrap();
            builder.append(true).unwrap();
            builder.append_null().unwrap();
            builder.append(true).unwrap();
            builder.append(true).unwrap();

            builder.finish()
        };

        assert!(a.equals(&b));
        assert!(b.equals(&a));
    }
}
