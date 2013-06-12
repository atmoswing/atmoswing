#include "asMethodCalibratorClassicPlusVarExplo.h"

asMethodCalibratorClassicPlusVarExplo::asMethodCalibratorClassicPlusVarExplo()
:
asMethodCalibratorClassicPlus()
{

}

asMethodCalibratorClassicPlusVarExplo::~asMethodCalibratorClassicPlusVarExplo()
{

}

bool asMethodCalibratorClassicPlusVarExplo::Calibrate(asParametersCalibration &params)
{

    int i_step;
    wxFileConfig::Get()->Read("/Calibration/VariablesExplo/Step", &i_step, params.GetStepsNb()-1);

    asLogMessageImportant(wxString::Format(_("Processing variables exploration for step %d"), i_step));
    asLogMessageImportant(wxString::Format(_("Processing %d variables, %d hours, %d levels, %d criteria."),
                                           (int)params.GetPredictorDataIdVector(i_step, 0).size(),
                                           (int)params.GetPredictorDTimeHoursVector(i_step, 0).size(),
                                           (int)params.GetPredictorLevelVector(i_step, 0).size(),
                                           (int)params.GetPredictorCriteriaVector(i_step, 0).size()));

    if (i_step>=params.GetStepsNb())
    {
        asLogError(_("The given step number for variables exploration is above available steps."));
        return false;
    }

    for(int i_ptor=0; i_ptor<params.GetPredictorsNb(i_step); i_ptor++)
    {
        if(params.NeedsPreprocessing(i_step, i_ptor))
        {

        asLogError(_("Calibration method not implemented to work with preprocessed data."));
        return false;
        }
        else
        {
            VectorString vPredictorDataId = params.GetPredictorDataIdVector(i_step, i_ptor);

            for(unsigned int i_predictordata=0; i_predictordata<vPredictorDataId.size(); i_predictordata++)
            {
                params.SetPredictorDataId(i_step, i_ptor, vPredictorDataId[i_predictordata]);

                VectorDouble vPredictorDTimeHours = params.GetPredictorDTimeHoursVector(i_step, i_ptor);

                for(unsigned int i_predictortime=0; i_predictortime<vPredictorDTimeHours.size(); i_predictortime++)
                {
                    params.SetPredictorDTimeHours(i_step, i_ptor, vPredictorDTimeHours[i_predictortime]);
                    params.FixTimeShift();

                    VectorFloat vPredictorLevels = params.GetPredictorLevelVector(i_step, i_ptor);

                    for(unsigned int i_predictorlevel=0; i_predictorlevel<vPredictorLevels.size(); i_predictorlevel++)
                    {
                        params.SetPredictorLevel(i_step, i_ptor, vPredictorLevels[i_predictorlevel]);

                        VectorString vPredictorCriteria = params.GetPredictorCriteriaVector(i_step, i_ptor);

                        for(unsigned int i_criteria=0; i_criteria<vPredictorCriteria.size(); i_criteria++)
                        {
                            params.SetPredictorCriteria(i_step, i_ptor, vPredictorCriteria[i_criteria]);

                            VectorFloat slctPredictorLevels;
                            slctPredictorLevels.push_back(vPredictorLevels[i_predictorlevel]);
                            params.SetPreloadLevels(i_step, i_ptor, slctPredictorLevels);

                            VectorDouble slctPreloadDTimeHours;
                            slctPreloadDTimeHours.push_back(vPredictorDTimeHours[i_predictortime]);
                            params.SetPreloadDTimeHours(i_step, i_ptor, slctPreloadDTimeHours);

                            m_OriginalParams = params;

                            if(!asMethodCalibratorClassicPlus::Calibrate(params)) return false;

                            params = m_OriginalParams;

                            ClearAll();
                        }
                    }
                }
            }
        }
    }

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}
