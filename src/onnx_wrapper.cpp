// ONNX Wrapper - instantiates ONNX types to pull in symbols
// This avoids the need for whole-archive linking

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>
#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>

#include <cstdint>
#include <cmath>

// Force instantiation of commonly used ONNX types and functions
// by creating a dummy function that uses them
extern "C" {

// Version info
const char* onnx_wrapper_version() {
    return "0.1.0";
}

// Dummy function that references all the types we need
// This ensures the linker includes them in the shared library
void onnx_wrapper_instantiate_types() {
    // Model level
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.set_producer_name("onnx-wrapper");
    model.set_producer_version("0.1.0");
    model.set_model_version(1);
    model.set_doc_string("dummy");

    // Opset imports
    auto* opset = model.add_opset_import();
    opset->set_version(18);
    opset->set_domain("");

    // Graph
    auto* graph = model.mutable_graph();
    graph->set_name("dummy_graph");

    // Graph inputs
    auto* input = graph->add_input();
    input->set_name("input");
    input->set_doc_string("input doc");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(onnx::TensorProto::DOUBLE);
    auto* input_shape = input_type->mutable_shape();
    input_shape->add_dim()->set_dim_value(1);
    input_shape->add_dim()->set_dim_param("N");

    // Graph outputs
    auto* output = graph->add_output();
    output->set_name("output");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(onnx::TensorProto::DOUBLE);
    output_type->mutable_shape()->add_dim()->set_dim_value(1);

    // Metadata
    auto* meta = input->add_metadata_props();
    meta->set_key("key");
    meta->set_value("value");

    // Nodes with various op types
    auto* node = graph->add_node();
    node->set_op_type("Add");
    node->set_name("add_node");
    node->set_domain("");
    node->set_doc_string("node doc");
    node->add_input("a");
    node->add_input("b");
    node->add_output("c");

    // Node metadata
    auto* node_meta = node->add_metadata_props();
    node_meta->set_key("node_key");
    node_meta->set_value("node_value");

    // Attributes - various types
    auto* attr_int = node->add_attribute();
    attr_int->set_name("axis");
    attr_int->set_type(onnx::AttributeProto::INT);
    attr_int->set_i(0);

    auto* attr_float = node->add_attribute();
    attr_float->set_name("alpha");
    attr_float->set_type(onnx::AttributeProto::FLOAT);
    attr_float->set_f(1.0f);

    auto* attr_string = node->add_attribute();
    attr_string->set_name("mode");
    attr_string->set_type(onnx::AttributeProto::STRING);
    attr_string->set_s("constant");

    auto* attr_ints = node->add_attribute();
    attr_ints->set_name("pads");
    attr_ints->set_type(onnx::AttributeProto::INTS);
    attr_ints->add_ints(0);
    attr_ints->add_ints(0);

    auto* attr_floats = node->add_attribute();
    attr_floats->set_name("scales");
    attr_floats->set_type(onnx::AttributeProto::FLOATS);
    attr_floats->add_floats(1.0f);

    // Tensor attribute
    auto* attr_tensor = node->add_attribute();
    attr_tensor->set_name("value");
    attr_tensor->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr_tensor->mutable_t();
    tensor->set_name("const_tensor");
    tensor->set_data_type(onnx::TensorProto::DOUBLE);
    tensor->add_dims(2);
    tensor->add_dims(3);
    tensor->add_double_data(1.0);
    tensor->add_double_data(2.0);

    // Other tensor data types
    onnx::TensorProto int_tensor;
    int_tensor.set_data_type(onnx::TensorProto::INT64);
    int_tensor.add_int64_data(42);

    onnx::TensorProto bool_tensor;
    bool_tensor.set_data_type(onnx::TensorProto::BOOL);
    bool_tensor.add_int32_data(1);

    onnx::TensorProto float_tensor;
    float_tensor.set_data_type(onnx::TensorProto::FLOAT);
    float_tensor.add_float_data(3.14f);

    // Graph attribute (for Loop, If, etc.)
    auto* attr_graph = node->add_attribute();
    attr_graph->set_name("body");
    attr_graph->set_type(onnx::AttributeProto::GRAPH);
    auto* subgraph = attr_graph->mutable_g();
    subgraph->set_name("loop_body");

    // Subgraph inputs/outputs
    auto* sub_input = subgraph->add_input();
    sub_input->set_name("i");
    sub_input->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::INT64);

    auto* sub_output = subgraph->add_output();
    sub_output->set_name("cond");
    sub_output->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::BOOL);

    // Initializers
    auto* initializer = graph->add_initializer();
    initializer->set_name("weights");
    initializer->set_data_type(onnx::TensorProto::DOUBLE);
    initializer->add_dims(10);
    for (int i = 0; i < 10; i++) {
        initializer->add_double_data(static_cast<double>(i));
    }

    // Functions
    auto* func = model.add_functions();
    func->set_name("custom_op");
    func->set_domain("custom");
    func->add_input("x");
    func->add_output("y");

    auto* func_opset = func->add_opset_import();
    func_opset->set_version(18);

    auto* func_node = func->add_node();
    func_node->set_op_type("Relu");
    func_node->add_input("x");
    func_node->add_output("y");

    // ValueInfo operations
    onnx::ValueInfoProto value_info;
    value_info.set_name("tensor_info");
    value_info.mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::DOUBLE);

    // TypeProto operations
    onnx::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(onnx::TensorProto::FLOAT);

    // TensorShapeProto operations
    onnx::TensorShapeProto shape_proto;
    shape_proto.add_dim()->set_dim_value(1);
    shape_proto.add_dim()->set_dim_param("batch");

    // Copy operations
    onnx::NodeProto node_copy;
    node_copy.CopyFrom(*node);

    onnx::TensorProto tensor_copy;
    tensor_copy.CopyFrom(*tensor);

    onnx::GraphProto graph_copy;
    graph_copy.CopyFrom(*graph);

    // Serialization
    std::string serialized;
    model.SerializeToString(&serialized);

    onnx::ModelProto parsed;
    parsed.ParseFromString(serialized);

    // Iteration over collections
    for (int i = 0; i < graph->node_size(); i++) {
        const auto& n = graph->node(i);
        (void)n.op_type();
        (void)n.name();
    }

    for (int i = 0; i < graph->input_size(); i++) {
        const auto& inp = graph->input(i);
        (void)inp.name();
    }

    for (int i = 0; i < graph->output_size(); i++) {
        const auto& out = graph->output(i);
        (void)out.name();
    }

    for (int i = 0; i < graph->initializer_size(); i++) {
        const auto& init = graph->initializer(i);
        (void)init.name();
    }
}

// Wrapper for shape inference
int onnx_wrapper_infer_shapes(void* model_ptr) {
    if (!model_ptr) return -1;
    try {
        onnx::ModelProto* model = static_cast<onnx::ModelProto*>(model_ptr);
        onnx::shape_inference::InferShapes(*model);
        return 0;
    } catch (...) {
        return -1;
    }
}

// Wrapper for model checking
int onnx_wrapper_check_model(void* model_ptr) {
    if (!model_ptr) return -1;
    try {
        onnx::ModelProto* model = static_cast<onnx::ModelProto*>(model_ptr);
        onnx::checker::check_model(*model, false, false, false);
        return 0;
    } catch (const onnx::checker::ValidationError&) {
        return -1;
    } catch (...) {
        return -2;
    }
}

} // extern "C"

// Additional symbol instantiation to match CasADi usage patterns
namespace {

// Exercise TensorShapeProto dimension access (used heavily in CasADi import)
void force_shape_operations() {
    onnx::TensorShapeProto shape;

    // Add dimensions
    auto* dim1 = shape.add_dim();
    dim1->set_dim_value(10);

    auto* dim2 = shape.add_dim();
    dim2->set_dim_param("batch_size");

    // Access dimensions
    int num_dims = shape.dim_size();
    for (int i = 0; i < num_dims; ++i) {
        const auto& dim = shape.dim(i);

        if (dim.has_dim_value()) {
            int64_t val = dim.dim_value();
            (void)val;
        }

        if (dim.has_dim_param()) {
            const std::string& param = dim.dim_param();
            (void)param;
        }
    }

    // Mutable access
    auto* mutable_dim = shape.mutable_dim(0);
    mutable_dim->clear_dim_value();
    mutable_dim->set_dim_param("N");
}

// Exercise TypeProto (the specific symbol mentioned: _TypeProto_default_instance_)
void force_type_proto_operations() {
    onnx::TypeProto type;

    // Tensor type
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(onnx::TensorProto::DOUBLE);

    auto* shape = tensor_type->mutable_shape();
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(4);

    // Access
    if (type.has_tensor_type()) {
        const auto& tt = type.tensor_type();
        int elem_type = tt.elem_type();
        (void)elem_type;

        if (tt.has_shape()) {
            const auto& s = tt.shape();
            (void)s.dim_size();
        }
    }

    // TypeProto::Tensor nested type
    onnx::TypeProto::Tensor standalone_tensor_type;
    standalone_tensor_type.set_elem_type(onnx::TensorProto::FLOAT);

    // Clear and copy
    onnx::TypeProto type_copy;
    type_copy.CopyFrom(type);
    type.Clear();
}

// Exercise TensorProto data access (all data types used by CasADi)
void force_tensor_data_operations() {
    // DOUBLE tensor
    onnx::TensorProto double_tensor;
    double_tensor.set_name("double_data");
    double_tensor.set_data_type(onnx::TensorProto::DOUBLE);
    double_tensor.add_dims(2);
    double_tensor.add_dims(3);
    double_tensor.add_double_data(1.0);
    double_tensor.add_double_data(2.0);

    // Access double data
    int double_size = double_tensor.double_data_size();
    for (int i = 0; i < double_size; ++i) {
        double val = double_tensor.double_data(i);
        (void)val;
    }

    // FLOAT tensor
    onnx::TensorProto float_tensor;
    float_tensor.set_data_type(onnx::TensorProto::FLOAT);
    float_tensor.add_float_data(3.14f);
    int float_size = float_tensor.float_data_size();
    for (int i = 0; i < float_size; ++i) {
        float val = float_tensor.float_data(i);
        (void)val;
    }

    // INT32 tensor
    onnx::TensorProto int32_tensor;
    int32_tensor.set_data_type(onnx::TensorProto::INT32);
    int32_tensor.add_int32_data(42);
    int int32_size = int32_tensor.int32_data_size();
    for (int i = 0; i < int32_size; ++i) {
        int32_t val = int32_tensor.int32_data(i);
        (void)val;
    }

    // INT64 tensor
    onnx::TensorProto int64_tensor;
    int64_tensor.set_data_type(onnx::TensorProto::INT64);
    int64_tensor.add_int64_data(123456789LL);
    int int64_size = int64_tensor.int64_data_size();
    for (int i = 0; i < int64_size; ++i) {
        int64_t val = int64_tensor.int64_data(i);
        (void)val;
    }

    // BOOL tensor (stored in int32_data)
    onnx::TensorProto bool_tensor;
    bool_tensor.set_data_type(onnx::TensorProto::BOOL);
    bool_tensor.add_int32_data(1);

    // Raw data access
    onnx::TensorProto raw_tensor;
    raw_tensor.set_data_type(onnx::TensorProto::DOUBLE);
    double raw_values[] = {1.0, 2.0, 3.0};
    raw_tensor.set_raw_data(raw_values, sizeof(raw_values));

    if (raw_tensor.has_raw_data()) {
        const std::string& raw = raw_tensor.raw_data();
        size_t raw_size = raw.size();
        (void)raw_size;
    }

    // Dims access
    int dims_size = double_tensor.dims_size();
    for (int i = 0; i < dims_size; ++i) {
        int64_t dim = double_tensor.dims(i);
        (void)dim;
    }
}

// Exercise ValueInfoProto (used for graph inputs/outputs)
void force_value_info_operations() {
    onnx::ValueInfoProto value_info;
    value_info.set_name("tensor_value");
    value_info.set_doc_string("A tensor value");

    // Access type
    auto* type = value_info.mutable_type();
    auto* tensor_type = type->mutable_tensor_type();
    tensor_type->set_elem_type(onnx::TensorProto::DOUBLE);

    auto* shape = tensor_type->mutable_shape();
    shape->add_dim()->set_dim_value(10);

    // Read back
    const std::string& name = value_info.name();
    (void)name;

    if (value_info.has_type()) {
        const auto& t = value_info.type();
        if (t.has_tensor_type()) {
            const auto& tt = t.tensor_type();
            (void)tt.elem_type();
        }
    }
}

// Exercise OperatorSetIdProto (opset imports)
void force_opset_operations() {
    onnx::OperatorSetIdProto opset;
    opset.set_domain("");
    opset.set_version(13);

    const std::string& domain = opset.domain();
    int64_t version = opset.version();
    (void)domain; (void)version;
}

// Exercise FunctionProto (local functions in ONNX)
void force_function_proto_operations() {
    onnx::FunctionProto func;
    func.set_name("my_function");
    func.set_domain("custom_domain");

    // Add inputs/outputs
    func.add_input("x");
    func.add_input("y");
    func.add_output("z");

    // Add opset import
    auto* opset = func.add_opset_import();
    opset->set_version(13);

    // Add a node
    auto* node = func.add_node();
    node->set_op_type("Add");
    node->add_input("x");
    node->add_input("y");
    node->add_output("z");

    // Access
    int input_size = func.input_size();
    int output_size = func.output_size();
    int node_size = func.node_size();

    for (int i = 0; i < input_size; ++i) {
        const std::string& inp = func.input(i);
        (void)inp;
    }

    for (int i = 0; i < output_size; ++i) {
        const std::string& out = func.output(i);
        (void)out;
    }

    for (int i = 0; i < node_size; ++i) {
        const auto& n = func.node(i);
        (void)n.op_type();
    }

    (void)input_size; (void)output_size; (void)node_size;
}

// Exercise file I/O operations (ParseFromIstream, SerializeToOstream)
void force_io_operations() {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.set_producer_name("test");

    // Serialize to string
    std::string serialized;
    model.SerializeToString(&serialized);

    // Parse from string
    onnx::ModelProto parsed;
    parsed.ParseFromString(serialized);

    // Byte size
    size_t byte_size = model.ByteSizeLong();
    (void)byte_size;

    // Is initialized
    bool initialized = model.IsInitialized();
    (void)initialized;

    // DebugString (useful for debugging)
    std::string debug = model.DebugString();
    (void)debug;
}

// Exercise StringStringEntryProto (metadata)
void force_metadata_operations() {
    onnx::StringStringEntryProto entry;
    entry.set_key("key");
    entry.set_value("value");

    const std::string& key = entry.key();
    const std::string& value = entry.value();
    (void)key; (void)value;

    // In model metadata
    onnx::ModelProto model;
    auto* meta = model.add_metadata_props();
    meta->set_key("author");
    meta->set_value("CasADi");

    int meta_size = model.metadata_props_size();
    for (int i = 0; i < meta_size; ++i) {
        const auto& m = model.metadata_props(i);
        (void)m.key();
    }
}

// Force instantiation at static initialization time
struct StaticInitializer {
    StaticInitializer() {
        // These are called at load time to ensure symbols are linked
        // They won't actually run unless the library is loaded
    }
} static_init;

} // anonymous namespace
