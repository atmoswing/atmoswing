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
        int archiveMembersNb;
        wxString realtimeDatasetId;
        wxString realtimeDataId;
        int realtimeMembersNb;
        vwxs preprocessArchiveDatasetIds;
        vwxs preprocessArchiveDataIds;
        int preprocessArchiveMembersNb;
        vwxs preprocessRealtimeDatasetIds;
        vwxs preprocessRealtimeDataIds;
        int preprocessRealtimeMembersNb;
    } ParamsPredictorForecast;

    typedef std::vector<ParamsPredictorForecast> VectorParamsPredictorsForecast;

    typedef struct
    {
        vi analogsNumberLeadTime;
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

    bool SetLeadTimeDaysVector(vi val);

    vi GetLeadTimeDaysVector() const
    {
        return m_leadTimeDaysVect;
    }

    int GetLeadTimeDays(int iLead) const
    {
        return m_leadTimeDaysVect[iLead];
    }

    int GetLeadTimeHours(int iLead) const
    {
        return (int) (m_leadTimeDaysVect[iLead] * 24.0);
    }

    bool SetAnalogsNumberLeadTimeVector(int iStep, vi val);

    vi GetAnalogsNumberLeadTimeVector(int iStep) const
    {
        return m_stepsForecast[iStep].analogsNumberLeadTime;
    }

    int GetAnalogsNumberLeadTime(int iStep, int iLead) const
    {
        wxASSERT((int) m_stepsForecast[iStep].analogsNumberLeadTime.size() > iLead);
        return m_stepsForecast[iStep].analogsNumberLeadTime[iLead];
    }

    wxString GetPredictorArchiveDatasetId(int iStep, int iPtor) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].archiveDatasetId;
    }

    bool SetPredictorArchiveDatasetId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorArchiveDataId(int iStep, int iPtor) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].archiveDataId;
    }

    bool SetPredictorArchiveDataId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorRealtimeDatasetId(int iStep, int iPtor) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeDatasetId;
    }

    bool SetPredictorRealtimeDatasetId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorRealtimeDataId(int iStep, int iPtor) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeDataId;
    }

    bool SetPredictorRealtimeDataId(int iStep, int iPtor, const wxString &val);

    int GetPreprocessSize(int iStep, int iPtor) const
    {
        return (int) m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.size();
    }

    wxString GetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessArchiveDataId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessArchiveDataId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre, const wxString &val);

    void SetPredictorArchiveMembersNb(int iStep, int iPtor, int val)
    {
        m_stepsForecast[iStep].predictors[iPtor].archiveMembersNb = val;
    }

    int GetPredictorArchiveMembersNb(int iStep, int iPtor) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].archiveMembersNb;
    }

    void SetPredictorRealtimeMembersNb(int iStep, int iPtor, int val)
    {
        m_stepsForecast[iStep].predictors[iPtor].realtimeMembersNb = val;
    }

    int GetPredictorRealtimeMembersNb(int iStep, int iPtor) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeMembersNb;
    }

    void SetPreprocessArchiveMembersNb(int iStep, int iPtor, int iPre, int val)
    {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveMembersNb = val;
    }

    int GetPreprocessArchiveMembersNb(int iStep, int iPtor, int iPre) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveMembersNb;
    }

    void SetPreprocessRealtimeMembersNb(int iStep, int iPtor, int iPre, int val)
    {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeMembersNb = val;
    }

    int GetPreprocessRealtimeMembersNb(int iStep, int iPtor, int iPre) const
    {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeMembersNb;
    }

protected:

private:
    vi m_leadTimeDaysVect;
    VectorParamsStepForecast m_stepsForecast;
    wxString m_predictandDatabase;

    bool ParseDescription(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersForecast &fileParams, int iStep, const wxXmlNode *nodeProcess);

    bool ParsePreprocessedPredictors(asFileParametersForecast &fileParams, int iStep, int iPtor,
                                     const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess);
};

#endif // ASPARAMETERSFORECAST_H
