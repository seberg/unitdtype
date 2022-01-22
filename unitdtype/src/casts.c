#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "casts.h"
#include "dtype.h"


/*
 * NOTES
 * -----
 * The current implementation here used `unyt.Unit.get_conversion_factor()`.
 * But wraps it into our little Python helper to have a fast LRU cache.
 * Internally, the units actually have a `base_offset` each, so I wonder if
 * conversion may not be better to be done using the formula:
 *
 *      converted = factor * (old - base_offset_old) + base_offset
 *
 * (or ratio as a word instead of factor).  Rather than finding a single offset
 * here.  (I assume speed is irrelevant, because offsets are rare.)
 */
int
get_conversion_factor(
        PyObject *from_unit, PyObject *to_unit, double *factor, double *offset)
{
    static PyObject *get_conversion_factor = NULL;

    if (from_unit == to_unit) {
        *factor = 1;
        *offset = 0;
        return 0;
    }

    if (NPY_UNLIKELY(get_conversion_factor) == NULL) {
        PyObject *mod = PyImport_ImportModule("unitdtype._helpers");
        if (mod == NULL) {
            return NULL;
        }
        get_conversion_factor = PyObject_GetAttrString(
                mod, "get_conversion_factor");
        Py_DECREF(mod);
        if (get_conversion_factor == NULL) {
            return NULL;
        }
    }

    PyObject *conv = PyObject_CallFunctionObjArgs(
            get_conversion_factor, from_unit, to_unit, NULL);
    if (conv == NULL) {
        return -1;
    }
    if (!PyTuple_CheckExact(conv) || PyTuple_GET_SIZE(conv) != 2) {
        PyErr_SetString(PyExc_TypeError,
                "unit error, conversion was not a tuple.");
        return -1;
    }
    // TODO: Do I need to check types better?
    *factor = PyFloat_AsDouble(PyTuple_GET_ITEM(conv, 0));\
    if (*factor == -1 && PyErr_Occurred()) {
        Py_DECREF(conv);
        return -1;
    }
    *offset = 0;
    if (PyTuple_GET_ITEM(conv, 1) != Py_None) {
        *offset = PyFloat_AsDouble(PyTuple_GET_ITEM(conv, 1));
        if (*factor == -1 && PyErr_Occurred()) {
            Py_DECREF(conv);
            return -1;
        }
    }
    Py_DECREF(conv);
    return 0;
}


static NPY_CASTING
unit_to_unit_resolve_descriptors(
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    double factor = 1., offset = 0.;
    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        // TODO: We could add `auxdata` already here.  That is a bit useless
        //       for can-cast, but would safe us from finding the factor again
        //       later.  Alternatively, we could allow auxdata=NULL disallowing
        //       passing of auxdata.  With the promise that it is never NULL
        //       if we follow up with using the loops.
        //       (That would also avoid forcing the use of get_loop or similar
        //       in the future, hmmmmm)
        if (get_conversion_factor(
                ((UnitDTypeObject *)given_descrs[0])->unit,
                ((UnitDTypeObject *)given_descrs[1])->unit,
                &factor, &offset) < 0) {
            return -1;
        }
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (factor == 1 && offset == 0) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }

    /* Should this use safe casting? */
    return NPY_SAME_KIND_CASTING;
}


typedef struct {
    NpyAuxData base;
    double factor;
    double offset;
} conv_auxdata;


#define CONV_AUXDATA_FREELIST_SIZE 5
static int conv_auxdata_freenum = 0;
static conv_auxdata *conv_auxdata_freelist[CONV_AUXDATA_FREELIST_SIZE] = {NULL};


static void
conv_auxdata_free(conv_auxdata *conv_auxdata)
{
    if (conv_auxdata_freenum < CONV_AUXDATA_FREELIST_SIZE) {
        conv_auxdata_freelist[conv_auxdata_freenum] = conv_auxdata;
    }
    else {
        PyMem_Free(conv_auxdata);
    }
}


static conv_auxdata *
get_conv_factor_auxdata(UnitDTypeObject *from_dt, UnitDTypeObject *to_dt)
{
    conv_auxdata *res;
    if (conv_auxdata_freenum > 0) {
        conv_auxdata_freenum--;
        res = conv_auxdata_freelist[conv_auxdata_freenum];
    }
    else {
        res = PyMem_Calloc(1, sizeof(conv_auxdata));
        if (res < 0) {
            PyErr_NoMemory();
            return NULL;
        }
        res->base.free = (void *)conv_auxdata_free;
    }
    if (get_conversion_factor(
            from_dt->unit, to_dt->unit, &res->factor, &res->offset) < 0) {
        NPY_AUXDATA_FREE((NpyAuxData *)res);
        return NULL;
    }
    return res;
}


static NPY_GCC_OPT_3 NPY_GCC_UNROLL_LOOPS int
unit_to_unit_contiguous_no_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const NPY_UNUSED(strides[]), conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    npy_intp N = dimensions[0];
    double *in = (double *)data[0];
    double *out = (double *)data[1];

    while (N--) {
        *out = factor * *in;
        out++;
        in++;
    }
    return 0;
}


static NPY_GCC_OPT_3 NPY_GCC_UNROLL_LOOPS int
unit_to_unit_strided_no_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        *(double *)out = factor * *(double *)in;
        in += in_stride;
        out += out_stride;
    }
    return 0;
}


static NPY_GCC_OPT_3 NPY_GCC_UNROLL_LOOPS int
unit_to_unit_contiguous_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const NPY_UNUSED(strides[]), conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    double offset = auxdata->offset;
    npy_intp N = dimensions[0];
    double *in = (double *)data[0];
    double *out = (double *)data[1];

    while (N--) {
        *out = factor * (*in + offset);
        out++;
        in++;
    }
    return 0;
}


static NPY_GCC_OPT_3 NPY_GCC_UNROLL_LOOPS int
unit_to_unit_strided_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    double offset = auxdata->offset;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        *(double *)out = factor * (*(double *)in + offset);
        in += in_stride;
        out += out_stride;
    }
    return 0;
}


static NPY_GCC_OPT_3 NPY_GCC_UNROLL_LOOPS int
unit_to_unit_unaligned(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    double offset = auxdata->offset;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        double in_val, out_val;
        memcpy(&in_val, in, sizeof(double));
        out_val = factor * (in_val + offset);
        memcpy(out, &out_val, sizeof(double));
        in += in_stride;
        out += out_stride;
    }
    return 0;
}


// TODO: This still needs to make public officially.  I don't like the API
//       but OTOH, wrapping things into that API is probably OKish...
//       (i.e. we can still introduce better API, nothing stopping us :))
static int
unit_to_unit_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    conv_auxdata *conv_auxdata = get_conv_factor_auxdata(
            (UnitDTypeObject *)context->descriptors[0],
            (UnitDTypeObject *)context->descriptors[1]);
    if (conv_auxdata == NULL) {
        return -1;
    }
    *out_transferdata = (NpyAuxData *)conv_auxdata;

    if (aligned) {
        int contig = (strides[0] == sizeof(double)
                      && strides[1] == sizeof(double));
        if (contig) {
            if (conv_auxdata->offset == 0) {
                *out_loop = (PyArrayMethod_StridedLoop *)&unit_to_unit_contiguous_no_offset;
            }
            else {
                *out_loop = (PyArrayMethod_StridedLoop *)&unit_to_unit_contiguous_offset;
            }
        }
        else {
            if (conv_auxdata->offset == 0) {
                *out_loop = (PyArrayMethod_StridedLoop *)&unit_to_unit_strided_no_offset;
            }
            else {
                *out_loop = (PyArrayMethod_StridedLoop *)&unit_to_unit_strided_offset;
            }
        }
    }
    else {
        *out_loop = (PyArrayMethod_StridedLoop *)&unit_to_unit_unaligned;
    }

    *flags = 0;
    return 0;
}


static PyObject *dtypes[2] = {NULL, NULL};

static PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, &unit_to_unit_resolve_descriptors},
        {_NPY_METH_get_loop, &unit_to_unit_get_loop},
        {0, NULL}
};


PyArrayMethod_Spec UnitToUnitCastSpec = {
    .name = "cast_Float64Unit_to_Float64Unit",
    .nin = 1,
    .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED,
    .casting = NPY_SAME_KIND_CASTING,
    .dtypes = dtypes,
    .slots = slots,
};
