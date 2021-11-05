#include "Python.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pybind11/cast.h>
#include <pybind11/attr.h>

#include <pdal/io/MemoryViewReader.hpp>
#include <pdal/pdal_features.hpp>
#include <pdal/PipelineExecutor.hpp>

#include <iostream>
#include <numpy/arrayobject.h>
#include <arrow/python/pyarrow.h>


//int arrow_success = arrow::py::import_pyarrow();

namespace py = pybind11;
using namespace pybind11::detail;
using namespace pybind11::literals;

namespace pdal {
    class Pipeline
    {
    public:
        Pipeline(){}
        virtual ~Pipeline(){}

        PyObject *buildNumpyDescription(PointViewPtr view) {
            // Build up a numpy dtype dictionary
            //
            // {'formats': ['f8', 'f8', 'f8', 'u2', 'u1', 'u1', 'u1', 'u1', 'u1',
            //              'f4', 'u1', 'u2', 'f8', 'u2', 'u2', 'u2'],
            // 'names': ['X', 'Y', 'Z', 'Intensity', 'ReturnNumber',
            //           'NumberOfReturns', 'ScanDirectionFlag', 'EdgeOfFlightLine',
            //           'Classification', 'ScanAngleRank', 'UserData',
            //           'PointSourceId', 'GpsTime', 'Red', 'Green', 'Blue']}
            //
            Dimension::IdList dims = view->dims();
            PyObject *names = PyList_New(dims.size());
            PyObject *formats = PyList_New(dims.size());
            for (size_t i = 0; i < dims.size(); ++i) {
                Dimension::Id id = dims[i];
                std::string name = view->dimName(id);
                npy_intp stride = view->dimSize(id);

                std::string kind;
                Dimension::BaseType b = Dimension::base(view->dimType(id));
                if (b == Dimension::BaseType::Unsigned)
                    kind = "u";
                else if (b == Dimension::BaseType::Signed)
                    kind = "i";
                else if (b == Dimension::BaseType::Floating)
                    kind = "f";
                else
                    throw pdal_error("Unable to map kind '" + kind +
                                     "' to PDAL dimension type");

                std::stringstream oss;
                oss << kind << stride;
                PyList_SetItem(names, i, PyUnicode_FromString(name.c_str()));
                PyList_SetItem(formats, i, PyUnicode_FromString(oss.str().c_str()));
            }
            PyObject *dtype_dict = PyDict_New();
            PyDict_SetItemString(dtype_dict, "names", names);
            PyDict_SetItemString(dtype_dict, "formats", formats);
            return dtype_dict;
        }

        PipelineExecutor* get_executor() {
            std::string json(
                    "{ \"pipeline\": [ { \"type\": \"readers.las\", \"filename\": \"/Users/chloet/Documents/pb11_ex_skb/Data/1.2-with-color.las\" } ] }"
            );
            PipelineExecutor* executor = new PipelineExecutor(json);
            executor->execute();
            return executor;
        }

        py::array array_from_views() {
            if (_import_array() < 0)
                throw pdal_error("couldn't import numpy.core.multiarray");
            PointViewPtr view = *(get_executor()->getManagerConst().views().cbegin());
            PyObject* dtype_dict = buildNumpyDescription(view);
            PyArray_Descr* dtype = nullptr;
            if (PyArray_DescrConverter(dtype_dict, &dtype) == NPY_FAIL)
                throw pdal_error("Unable to build numpy dtype");
            Py_XDECREF(dtype_dict);
            npy_intp size = view->size();
            PyArrayObject* array = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, dtype,
                                                                        1, &size, 0, nullptr, NPY_ARRAY_CARRAY,
                                                                        nullptr);
            DimTypeList types = view->dimTypes();
            for (PointId idx = 0; idx < view->size(); idx++)
                view->getPackedPoint(types, idx, (char*)PyArray_GETPTR1(array, idx));
            py::array arr = py::cast<py::array>((PyObject*)array);
            return arr;
        }
    };

    class PyPipeline : public Pipeline
    {
        using Pipeline::Pipeline;
    };




}

PYBIND11_MODULE(_core, m)
{
    py::class_<pdal::Pipeline, pdal::PyPipeline>(m, "Pipeline", py::dynamic_attr())
            .def(py::init<>())
            .def("array", &pdal::Pipeline::array_from_views);
}


//pdal::Dimension::Type pdalType(int t)
//{
//    using namespace pdal::Dimension;
//
//    switch (t)
//    {
//        case NPY_FLOAT32:
//            return Type::Float;
//        case NPY_FLOAT64:
//            return Type::Double;
//        case NPY_INT8:
//            return Type::Signed8;
//        case NPY_INT16:
//            return Type::Signed16;
//        case NPY_INT32:
//            return Type::Signed32;
//        case NPY_INT64:
//            return Type::Signed64;
//        case NPY_UINT8:
//            return Type::Unsigned8;
//        case NPY_UINT16:
//            return Type::Unsigned16;
//        case NPY_UINT32:
//            return Type::Unsigned32;
//        case NPY_UINT64:
//            return Type::Unsigned64;
//        default:
//            return Type::None;
//    }
//    assert(0);
//
//    return Type::None;
//}
//
//
//class Manager {
//public:
//    Manager() {}
//    ~Manager() {}
//
//    void readPipeline(std::string& json) {}
//    int64_t execute() {
//        return 1;
//    }
//};
//
//
//class PipelineExecutor {
//public:
//    PipelineExecutor(std::string const& json) : m_json(json), m_executed(false) {}
//    ~PipelineExecutor() {}
//
//    int64_t execute() {
//        m_manager.readPipeline(m_json);
//        int64_t count = m_manager.execute();
//        m_executed = true;
//        return count;
//    }
//
//    std::string m_json;
//    Manager m_manager;
//    bool m_executed;
//
//};
//
//class PyArrayIter
//{
//public:
//    PyArrayIter(const PyArrayIter&) = delete;
//    PyArrayIter() {};
//
//    PyArrayIter(PyArrayObject* np_array) {
//        m_iter = NpyIter_New(np_array,
//                             NPY_ITER_EXTERNAL_LOOP | NPY_ITER_READONLY | NPY_ITER_REFS_OK,
//                             NPY_KEEPORDER, NPY_NO_CASTING, NULL);
//        if (!m_iter)
//            throw std::runtime_error("Unable to create numpy iterator.");
//
//        char *itererr;
//        m_iterNext = NpyIter_GetIterNext(m_iter, &itererr);
//        if (!m_iterNext)
//        {
//            NpyIter_Deallocate(m_iter);
//            throw std::runtime_error(std::string("Unable to create numpy iterator: ") +
//                             itererr);
//        }
//        m_data = NpyIter_GetDataPtrArray(m_iter);
//        m_stride = NpyIter_GetInnerStrideArray(m_iter);
//        m_size = NpyIter_GetInnerLoopSizePtr(m_iter);
//        m_done = false;
//    }
//
//    ~PyArrayIter() {
//        NpyIter_Deallocate(m_iter);
//    }
//    PyArrayIter& operator++() {
//        if (m_done)
//            return *this;
//
//        if (--(*m_size))
//            *m_data += *m_stride;
//        else if (!m_iterNext(m_iter))
//            m_done = true;
//        return *this;
//    }
//    operator bool () const { return !m_done; }
//    char* operator*() const { return *m_data; }
//
//private:
//    NpyIter* m_iter;
//    NpyIter_IterNextFunc *m_iterNext;
//    char** m_data;
//    npy_intp* m_size;
//    npy_intp* m_stride;
//    bool m_done;
//
//};
//
//class ReadArray : public PipelineExecutor {
//public:
//    ReadArray(std::string& json) : PipelineExecutor(json) {}
//    virtual ~ReadArray() {}
//
//    py::array m_array;
//
//    void setArray(py::object array_obj) {
//        py::array array = array_obj.cast<py::array>();
//        py::print(array.ndim());
//        npy_intp ndims = PyArray_NDIM((PyArrayObject *) array_obj.ptr());
//        py::print(ndims);
//        pdal::PipelineManager *manager = new pdal::PipelineManager();
//        std::stringstream ss(std::string("{\"pipeline\": [ {\"type\": \"filters.tail\", \"count\": 100 } ]}"));
//        manager->readPipeline(ss);
//        std::vector<pdal::Stage *> roots = manager->roots();
//        if (roots.size() != 1)
//            throw std::runtime_error("filter pipeline must contain a single root stage");
//
//        pdal::Options options;
//        bool rowMajor = !(PyArray_FLAGS((PyArrayObject *) array_obj.ptr()) & NPY_ARRAY_F_CONTIGUOUS);
//        options.add("order",
//                    rowMajor ? pdal::MemoryViewReader::Order::RowMajor : pdal::MemoryViewReader::Order::ColumnMajor);
//        options.add("shape", pdal::MemoryViewReader::Shape(array.shape(0), array.shape(1), array.shape(2)));
//        pdal::Stage &s = manager->makeReader("", "readers.memoryview", options);
//        pdal::MemoryViewReader &r = dynamic_cast<pdal::MemoryViewReader &>(s);
//
//        PyArray_Descr *dtype = PyArray_DTYPE((PyArrayObject *) array_obj.ptr());
//        int numFields = (dtype->fields == Py_None) ?
//                        0 :
//                        static_cast<int>(PyDict_Size(dtype->fields));
//        int xyz = 0;
//        if (numFields == 0) {
//            if (ndims != 3)
//                throw std::runtime_error("Array without fields must have 3 dimensions.");
//            r.pushField({"Intensity", pdalType(dtype->type_num), 0});
//        } else {
//            PyObject *names_dict = dtype->fields;
//            PyObject *names = PyDict_Keys(names_dict);
//            PyObject *values = PyDict_Values(names_dict);
//            if (!names || !values)
//                throw std::runtime_error("Bad field specification in numpy array.");
//
//            for (int i = 0; i < numFields; ++i) {
//                std::string name = py::str(PyList_GetItem(names, i));
//                if (name == "X")
//                    xyz |= 1;
//                else if (name == "Y")
//                    xyz |= 2;
//                else if (name == "Z")
//                    xyz |= 4;
//                PyObject *tup = PyList_GetItem(values, i);
//                size_t offset = PyLong_AsLong(PySequence_Fast_GET_ITEM(tup, 1));
//                PyArray_Descr *descr = (PyArray_Descr *) PySequence_Fast_GET_ITEM(tup, 0);
//                pdal::Dimension::Type type = pdalType(descr->type_num);
//                if (type == pdal::Dimension::Type::None)
//                    throw std::runtime_error("Incompatible type for field: " + name);
//                r.pushField({name, type, offset});
//            }
//
//            if (xyz != 0 && xyz != 7)
//                throw std::runtime_error("Array fields must contain all or none");
//            if (xyz == 0 && ndims != 3)
//                throw std::runtime_error("Array without named X/Y/Z must have 3 dims");
//        }
//
//        PyArrayIter* iter = new PyArrayIter();
//
////        auto incrementer = [&iter](pdal::PointId id) -> char * {
////            if (!iter)
////                return nullptr;
////            char *c = *iter;
////            ++iter;
////            return c;
////        };
//
////        r.setIncrementer(incrementer);
////        roots[0]->setInput(r);
//
//        m_array = array;
//    }
//
//    py::array getArray() {
//        return m_array;
//    }
//
//    void addArrayReader() {}
//};
//
//class PyReadArray : public ReadArray
//{
//public:
//    PyReadArray(std::string& json) : ReadArray(json) {}
//};
//
//
//PYBIND11_MODULE(_core, m) {
//    py::class_<PyArrayIter, std::shared_ptr<PyArrayIter>>(m, "PyArrayIter");
//    py::class_<ReadArray, PyReadArray>(m, "ReadArray", py::dynamic_attr())
//        .def(py::init<std::string &>())
//        .def_property("array", &ReadArray::getArray, &ReadArray::setArray)
//        .def("execute", &ReadArray::execute);
//}



//namespace pdal {
//
//int some_num() {
//    return 25;
//}
//
//class ReaderSTR;
//
//
//class Manager
//{
//public:
//    std::string manager_name;
//
//    void readString(std::string name_str);
//};
//
//
//class ReaderSTR
//{
//public:
//    ReaderSTR(Manager& manager) :  m_manager(manager) {}
//    void read(std::string name_str)
//    {
//        m_manager.manager_name = name_str;
//    }
//private:
//    Manager m_manager;
//};
//
//
//void Manager::readString(std::string name_str)
//{
//    ReaderSTR(*this).read(name_str);
//}
//
//class MyClass
//{
//private:
//    std::string name;
//    Manager m_manager;
//public:
//    MyClass(std::string i_name) : name(i_name), m_manager(Manager())
//    {}
//
//    void execute()
//    {
//        m_manager.readString(name);
//    }
//
//    std::string getName(){
//        return std::string("name is: ") + name;
//    }
//
//    void setName(std::string i_name)
//    {
//        name = i_name;
//    }
//};
//
//class MyClassShareable : public MyClass, public std::enable_shared_from_this<MyClassShareable>
//{
//public:
//    MyClassShareable(std::string i_name) : MyClass(i_name) {}
//};
//
//void doSomething(MyClassShareable* mc, std::string name){
//    mc->setName(name);
//}
//
//class FPC
//{
//public:
//    FPC(){}
//    virtual ~FPC(){}
//
//    std::shared_ptr<MyClass> _exec;
//    std::vector<py::array> _arrays;
//
//    std::string getOtherName(std::string i_name){
//        return _get_exec(i_name)->getName();
//    }
//
//    void execute(std::string i_name){
//        _get_exec(i_name)->execute();
//    }
//
//    void setArrays(std::vector<py::object> arrays){
//        _arrays.clear();
//        for (py::handle arr: arrays){
//            _arrays.push_back(py::cast<py::array>(arr));
//        }
//    }
//
//    py::object getArrays(){
//        char arr_kind = _arrays.at(0).dtype().kind();
//        if (arr_kind == 'i')
//            return getArraysT <int>();
//        else if (arr_kind == 'f')
//            return getArraysT <double>();
//        else
//            return py::object();
//    }
//
//    template<class T>
//    py::object getArraysT() {
//        std::vector<std::vector<T>> arrays;
//        for (py::handle arr: _arrays){
//            std::vector<T> arr_vec = py::cast<std::vector<T>>(arr);
//            arrays.push_back(arr_vec);
//        }
//        py::object arrays_obj = py::cast(&arrays);
//        return arrays_obj;
//    }
//
//    MyClass* _get_exec(std::string i_name){
//        if (!_exec)
//        {
//            _exec = std::make_shared<MyClass>(getSomeName());
//            addArrayReaders(_exec.get(), _arrays);
//        }
//        return _exec.get();
//    }
//
//    std::string getSomeName()
//    {
//        return "something or other";
//    }

//    void addArrayReaders(std::vector<std::shared_ptr<Array>> arrays)
//{
//    // Make the symbols in pdal_base global so that they're accessible
//    // to PDAL plugins.  Python dlopen's this extension with RTLD_LOCAL,
//    // which means that without this, symbols in libpdal_base aren't available
//    // for resolution of symbols on future runtime linking.  This is an issue
//    // on Alpine and other Linux variants that don't use UNIQUE symbols
//    // for C++ template statics only.  Without this, you end up with multiple
//    // copies of template statics.
//#ifndef _WIN32
//    ::dlopen("libpdal_base.so", RTLD_NOLOAD | RTLD_GLOBAL);
//#endif
//
//    PipelineManager& manager = getManager();
//    std::vector<Stage *> roots = manager.roots();
//    if (roots.size() != 1)
//        throw pdal_error("Filter pipeline must contain a single root stage.");
//
//    for (auto array : arrays)
//    {
//        // Create numpy reader for each array
//        // Options
//
//        Options options;
//        options.add("order", array->rowMajor() ?
//            MemoryViewReader::Order::RowMajor :
//            MemoryViewReader::Order::ColumnMajor);
//        options.add("shape", MemoryViewReader::Shape(array->shape()));
//
//        Stage& s = manager.makeReader("", "readers.memoryview", options);
//        MemoryViewReader& r = dynamic_cast<MemoryViewReader &>(s);
//        for (auto f : array->fields())
//            r.pushField(f);
//
//        ArrayIter& iter = array->iterator();
//        auto incrementer = [&iter](PointId id) -> char *
//        {
//            if (! iter)
//                return nullptr;
//
//            char *c = *iter;
//            ++iter;
//            return c;
//        };
//
//        r.setIncrementer(incrementer);
//        roots[0]->setInput(r);
//    }
//
//    manager.validateStageOptions();
//}

//};
//
//class PyFPC : public FPC
//{
//public:
//    using FPC::FPC;
//
//};
//}
