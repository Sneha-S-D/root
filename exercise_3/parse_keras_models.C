#include <TMVA/RModel.hxx>
#include <TMVA/RModelParser_Keras.hxx>

using namespace TMVA::Experimental::SOFIE;

void parse_keras_models() {
    RModelParser_Keras parser("model_dense.h5");
    RModel model = parser.Parse();
    model.Generate();
    model.OutputGenerated("model_dense.hxx");
}
