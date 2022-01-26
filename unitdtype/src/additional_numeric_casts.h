#ifndef _NPY_ADDITIONAL_NUMERIC_CASTS_H
#define _NPY_ADDITIONAL_NUMERIC_CASTS_H


/*
 * Verbose, but lets do it like this for now (there should be better ways to
 * do this), NumPy could help.  Another thing to consider is allowing
 * registering casts later, so that some of this could be done dynamically.
 */

PyArrayMethod_Spec BoolToUnitCastSpec;

PyArrayMethod_Spec UnitToUByteCastSpec;
PyArrayMethod_Spec UByteToUnitCastSpec;

PyArrayMethod_Spec UnitToUShortCastSpec;
PyArrayMethod_Spec UShortToUnitCastSpec;

PyArrayMethod_Spec UnitToUIntCastSpec;
PyArrayMethod_Spec UIntToUnitCastSpec;

PyArrayMethod_Spec UnitToULongCastSpec;
PyArrayMethod_Spec ULongToUnitCastSpec;

PyArrayMethod_Spec UnitToULongLongCastSpec;
PyArrayMethod_Spec ULongLongToUnitCastSpec;

PyArrayMethod_Spec UnitToByteCastSpec;
PyArrayMethod_Spec ByteToUnitCastSpec;

PyArrayMethod_Spec UnitToShortCastSpec;
PyArrayMethod_Spec ShortToUnitCastSpec;

PyArrayMethod_Spec UnitToIntCastSpec;
PyArrayMethod_Spec IntToUnitCastSpec;

PyArrayMethod_Spec UnitToLongCastSpec;
PyArrayMethod_Spec LongToUnitCastSpec;

PyArrayMethod_Spec UnitToLongLongCastSpec;
PyArrayMethod_Spec LongLongToUnitCastSpec;

PyArrayMethod_Spec UnitToFloatCastSpec;
PyArrayMethod_Spec FloatToUnitCastSpec;

#endif  /* _NPY_ADDITIONAL_NUMERIC_CASTS_H */