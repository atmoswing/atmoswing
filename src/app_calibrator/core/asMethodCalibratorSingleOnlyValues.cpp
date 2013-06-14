#include "asMethodCalibratorSingleOnlyValues.h"

asMethodCalibratorSingleOnlyValues::asMethodCalibratorSingleOnlyValues()
:
asMethodCalibrator()
{

}

asMethodCalibratorSingleOnlyValues::~asMethodCalibratorSingleOnlyValues()
{

}

bool asMethodCalibratorSingleOnlyValues::Calibrate(asParametersCalibration &params)
{
    // Check that we really handle a single case
    bool checkSizes = true;
    wxString errorField = wxEmptyString;
    if( params.GetTimeArrayAnalogsIntervalDaysVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("IntervalDays, ");
    }

    for (int i_step=0; i_step<params.GetStepsNb(); i_step++)
    {
        if( params.GetAnalogsNumberVector(i_step).size()>1 )
        {
            checkSizes = false;
            errorField.Append(wxString::Format("AnalogsNumber (step %d), ", i_step));
        }
        for (int i_predictor=0; i_predictor<params.GetPredictorsNb(i_step); i_predictor++)
        {
            if (params.NeedsPreprocessing(i_step, i_predictor))
            {
                for (int i_pre=0; i_pre<params.GetPreprocessSize(i_step, i_predictor); i_pre++)
                {
                    if( params.GetPreprocessDataIdVector(i_step, i_predictor, i_pre).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessDataId (step %d, predictor %d, preprocess %d), ", i_step, i_predictor, i_pre));
                    }
                    if( params.GetPreprocessLevelVector(i_step, i_predictor, i_pre).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessLevel (step %d, predictor %d, preprocess %d), ", i_step, i_predictor, i_pre));
                    }
                    if( params.GetPreprocessDTimeHoursVector(i_step, i_predictor, i_pre).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessDTimeHours (step %d, predictor %d, preprocess %d), ", i_step, i_predictor, i_pre));
                    }
                }
            }

            // Do the other ones anyway
            if( params.GetPredictorDataIdVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorDataId (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorLevelVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorLevel (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorDTimeHoursVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorDTimeHours (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorUminVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorUmin (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorUptsnbVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorUptsnb (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorVminVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorVmin (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorVptsnbVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorVptsnb (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorCriteriaVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorCriteria (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorWeightVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorWeight (step %d, predictor %d), ", i_step, i_predictor));
            }

            if(params.NeedsPreprocessing(i_step, i_predictor))
            {
                for (int i_dataset=0; i_dataset<params.GetPreprocessSize(i_step, i_predictor); i_dataset++)
                {
                    if( params.GetPreprocessLevelVector(i_step, i_predictor, i_dataset).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessLevel (step %d, predictor %d), ", i_step, i_predictor));
                    }
                    if( params.GetPreprocessDTimeHoursVector(i_step, i_predictor, i_dataset).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessDTimeHoursV (step %d, predictor %d), ", i_step, i_predictor));
                    }
                }
            }
        }
    }

    if(!checkSizes)
    {
        errorField = errorField.Remove(errorField.Length()-3, 2); // Removes the last coma
        wxString errorMessage = _("The following parameters are not compatible with the single assessment: ") + errorField;
        asLogError(errorMessage);
        return false;
    }

    // Extract the stations IDs
    VectorInt stationsId = params.GetPredictandStationsIdVector();

    // Create result object to save the final parameters sets
    asResultsParametersArray results_all;
    results_all.Init(_("all_station_parameters"));

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsAnalogsDates anaDatesPrevious;

    asLogMessageImportant(_("Do not process a score. Use to save intermediate values."));

    for (unsigned int i_stat=0; i_stat<stationsId.size(); i_stat++)
    {
        ClearAll();

        int stationId = stationsId[i_stat];

        // Reset the score of the climatology
        m_ScoreClimatology = 0;

        // Create results objects
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;

        // Set the next station ID
        params.SetPredictandStationId(stationId);

        // Process every step one after the other
        int stepsNb = params.GetStepsNb();
        for (int i_step=0; i_step<stepsNb; i_step++)
        {
            bool containsNaNs = false;
            if (i_step==0)
            {
                if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs)) return false;
            }
            else
            {
                if(!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs)) return false;
            }
            if (containsNaNs)
            {
                asLogError(_("The dates selection contains NaNs"));
                return false;
            }
            if(!GetAnalogsValues(anaValues, params, anaDates, i_step)) return false;

            // Keep the analogs dates of the best parameters set
            anaDatesPrevious = anaDates;
        }
    }

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}
