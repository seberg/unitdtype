#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H


typedef struct {
    PyArray_Descr base;
    PyObject *unit;
} UnitDTypeObject;

extern PyArray_DTypeMeta UnitDType_Type;

int
init_unit_dtype(void);

#endif  /*_NPY_DTYPE_H*/
