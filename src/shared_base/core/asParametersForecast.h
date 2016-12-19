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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASPARAMETERSFORECAST_H
#define ASPARAMETERSFORECAST_H

#include "asIncludes.h"
#include <asParameters.h>

class asFileParametersForecast;


class asParametersForecast
        : public asParameters
{
public:
    typedef struct
    {
        wxString archiveDatasetId;
        wxString archiveDataId;
        wxString realtimeDatasetId;
        wxString realtimeDataId;
        VectorString preprocessArchiveDatasetIds;
        VectorString preprocessArchiveDataIds;
        VectorString preprocessRealtimeDatasetIds;
        VectorString preprocessRealtimeDataIds;
    } ParamsPredictorForecast;

    typedef std::vector<ParamsPredictorForecast> VectorParamsPredictorsForecast;

    typedef struct
    {
        VectorInt analogsNumberLeadTime;
        VectorParamsPredictorsForecast predictors;
    } ParamsStepForecast;

    typedef std::vector<ParamsStepForecast> VectorParamsStepForecast;

    asParametersForecast();

    virtual ~asParametersForecast();

    void AddStep();

    void AddPredictorForecast(ParamsStepForecast &step);

    bool LoadFromFile(const wxString &filePath);

    bool InputsOK() const;

    void InitValues();

    wxString GetPredictandDatabase() const
    {
        return m_predictandDatabase;
    }

    void SetPredictandDatabase(wxString val)
    {
        m_predictandDatabase = val;
    }

    int GetLeadTimeNb() const
    {
        return (int) m_leadTimeDaysVect.size();
    }

    bool SetLeadTimeDaysVector(VectorInt val);

    VectorInt GetLeadTimeDaysVector() const
    {
        return m_leadTimeDaysVect;
    }

    int GetLeadTimeDays(int i_leadtime) const
    {
        return m_leadTimeDaysVect[i_leadtime];
    }

    int GetLeadTimeHours(int i_leadtime) const
    {
        return (int) (m_leadTimeDaysVect[i_leadtime] * 24.0);
    }

    bool SetAnalogsNumberLeadTimeVector(int i_step, VectorInt val);

    VectorInt GetAnalogsNumberLeadTimeVector(int i_step) const
    {
        return m_stepsForecast[i_step].analogsNumberLeadTime;
    }

    int GetAnalogsNumberLeadTime(int i_step, int i_leadtime) const
    {
        wxASSERT((int) m_stepsForecast[i_step].analogsNumberLeadTime.size() > i_leadtime);
        return m_stepsForecast[i_step].analogsNumberLeadTime[i_leadtime];
    }

    wxString GetPredictorArchiveDatasetId(int i_step, int i_predictor) const
    {
        return m_stepsForecast[i_step].predictors[i_predictor].archiveDatasetId;
    }

    bool SetPredictorArchiveDatasetId(int i_step, int i_predictor, const wxString &val);

    wxString GetPredictorArchiveDataId(int i_step, int i_predictor) const
    {
        return m_stepsForecast[i_step].predictors[i_predictor].archiveDataId;
    }

    bool SetPredictorArchiveDataId(int i_step, int i_predictor, const wxString &val);

    wxString GetPredictorRealtimeDatasetId(int i_step, int i_predictor) const
    {
        return m_stepsForecast[i_step].predictors[i_predictor].realtimeDatasetId;
    }

    bool SetPredictorRealtimeDatasetId(int i_step, int i_predictor, const wxString &val);

    wxString GetPredictorRealtimeDataId(int i_step, int i_predictor) const
    {
        return m_stepsForecast[i_step].predictors[i_predictor].realtimeDataId;
    }

    bool SetPredictorRealtimeDataId(int i_step, int i_predictor, const wxString &val);

    int GetPreprocessSize(int i_step, int i_predictor) const
    {
        return (int) m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDatasetIds.size();
    }

    wxString GetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset, const wxString &val);

    wxString GetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset, const wxString &val);

    wxString GetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset, const wxString &val);

    wxString GetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset, const wxString &val);

protected:

private:
    VectorInt m_leadTimeDaysVect;
    VectorParamsStepForecast m_stepsForecast;
    wxString m_predictandDatabase;

    bool ParseDescription(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersForecast &fileParams, int i_step, const wxXmlNode *nodeProcess);

    bool ParsePreprocessedPredictors(asFileParametersForecast &fileParams, int i_step, int i_ptor,
                                     const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess);
};

#endif // ASPARAMETERSFORECAST_H
