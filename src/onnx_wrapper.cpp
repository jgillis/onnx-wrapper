// ONNX Wrapper - instantiates ONNX types to pull in symbols
// This avoids the need for whole-archive linking

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>
#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>

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
