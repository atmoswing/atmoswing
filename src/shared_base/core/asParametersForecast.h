/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef ASPARAMETERSFORECAST_H
#define ASPARAMETERSFORECAST_H

#include "asIncludes.h"
#include <asParameters.h>

class asFileParametersForecast;


class asParametersForecast : public asParameters
{
public:
    // Structures
    typedef struct
    {
        wxString ArchiveDatasetId;
        wxString ArchiveDataId;
        wxString RealtimeDatasetId;
        wxString RealtimeDataId;
        VectorString PreprocessArchiveDatasetIds;
        VectorString PreprocessArchiveDataIds;
        VectorString PreprocessRealtimeDatasetIds;
        VectorString PreprocessRealtimeDataIds;
    } ParamsPredictorForecast;

    typedef std::vector < ParamsPredictorForecast > VectorParamsPredictorsForecast;

    typedef struct
    {
        VectorInt AnalogsNumberLeadTime;
        VectorParamsPredictorsForecast Predictors;
    } ParamsStepForecast;

    typedef std::vector < ParamsStepForecast > VectorParamsStepForecast;


    asParametersForecast();
    virtual ~asParametersForecast();

    void AddStep();

    void AddPredictorForecast(ParamsStepForecast &step);

    bool LoadFromFile(const wxString &filePath);

    bool InputsOK();

    void InitValues();

    wxString GetPredictandDatabase()
    {
        return m_predictandDatabase;
    }

    void SetPredictandDatabase(wxString val)
    {
        m_predictandDatabase = val;
    }

    int GetLeadTimeNb()
    {
        return m_leadTimeDaysVect.size();
    }

    bool SetLeadTimeDaysVector(VectorInt val);

    VectorInt GetLeadTimeDaysVector()
    {
        return m_leadTimeDaysVect;
    }

    int GetLeadTimeDays(int i_leadtime)
    {
        return m_leadTimeDaysVect[i_leadtime];
    }

    int GetLeadTimeHours(int i_leadtime)
    {
        return m_leadTimeDaysVect[i_leadtime]*24.0;
    }

    bool SetAnalogsNumberLeadTimeVector(int i_step, VectorInt val);

    VectorInt GetAnalogsNumberLeadTimeVector(int i_step)
    {
        return m_stepsForecast[i_step].AnalogsNumberLeadTime;
    }

    int GetAnalogsNumberLeadTime(int i_step, int i_leadtime)
    {
        wxASSERT(m_stepsForecast[i_step].AnalogsNumberLeadTime.size()>i_leadtime);
        return m_stepsForecast[i_step].AnalogsNumberLeadTime[i_leadtime];
    }

    int GetAnalogsNumberLeadTimeLastStep(int i_leadtime)
    {
        wxASSERT(m_stepsForecast[m_stepsForecast.size()-1].AnalogsNumberLeadTime.size()>i_leadtime);
        return m_stepsForecast[m_stepsForecast.size()-1].AnalogsNumberLeadTime[i_leadtime];
    }

    wxString GetPredictorArchiveDatasetId(int i_step, int i_predictor)
    {
        return m_stepsForecast[i_step].Predictors[i_predictor].ArchiveDatasetId;
    }

    bool SetPredictorArchiveDatasetId(int i_step, int i_predictor, const wxString& val);

    wxString GetPredictorArchiveDataId(int i_step, int i_predictor)
    {
        return m_stepsForecast[i_step].Predictors[i_predictor].ArchiveDataId;
    }

    bool SetPredictorArchiveDataId(int i_step, int i_predictor, const wxString& val);

    wxString GetPredictorRealtimeDatasetId(int i_step, int i_predictor)
    {
        return m_stepsForecast[i_step].Predictors[i_predictor].RealtimeDatasetId;
    }

    bool SetPredictorRealtimeDatasetId(int i_step, int i_predictor, const wxString& val);

    wxString GetPredictorRealtimeDataId(int i_step, int i_predictor)
    {
        return m_stepsForecast[i_step].Predictors[i_predictor].RealtimeDataId;
    }

    bool SetPredictorRealtimeDataId(int i_step, int i_predictor, const wxString& val);

    int GetPreprocessSize(int i_step, int i_predictor)
    {
        return m_stepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDatasetIds.size();
    }

    wxString GetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset, const wxString& val);

    wxString GetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset, const wxString& val);

    wxString GetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset, const wxString& val);

    wxString GetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset, const wxString& val);


protected:

private:
    VectorInt m_leadTimeDaysVect;
    VectorParamsStepForecast m_stepsForecast;
    wxString m_predictandDatabase;
};

#endif // ASPARAMETERSFORECAST_H
