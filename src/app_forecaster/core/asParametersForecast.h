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

#ifndef AS_PARAMETERS_FORECAST_H
#define AS_PARAMETERS_FORECAST_H

#include "asIncludes.h"
#include "asParameters.h"

class asFileParametersForecast;

class asParametersForecast : public asParameters {
  public:
    typedef struct ParamsPredictorForecast {
        wxString archiveDatasetId;
        wxString archiveDataId;
        bool archiveStandardize = false;
        double archiveStandardizeMean = NaNd;
        double archiveStandardizeSd = NaNd;
        int archiveMembersNb = 0;
        wxString realtimeDatasetId;
        wxString realtimeDataId;
        bool realtimeStandardize = false;
        double realtimeStandardizeMean = NaNd;
        double realtimeStandardizeSd = NaNd;
        int realtimeMembersNb = 0;
        vwxs preprocessArchiveDatasetIds;
        vwxs preprocessArchiveDataIds;
        int preprocessArchiveMembersNb = 0;
        vwxs preprocessRealtimeDatasetIds;
        vwxs preprocessRealtimeDataIds;
        int preprocessRealtimeMembersNb = 0;
    } ParamsPredictorForecast;

    typedef vector<ParamsPredictorForecast> VectorParamsPredictorsForecast;

    typedef struct ParamsStepForecast {
        vi analogsNumberLeadTime;
        VectorParamsPredictorsForecast predictors;
    } ParamsStepForecast;

    typedef vector<ParamsStepForecast> VectorParamsStepForecast;

    asParametersForecast();

    ~asParametersForecast() override;

    void AddStep() override;

    void AddPredictorForecast(ParamsStepForecast& step);

    bool LoadFromFile(const wxString& filePath) override;

    bool InputsOK() const override;

    void InitValues();

    wxString GetPredictandDatabase() const {
        return m_predictandDatabase;
    }

    void SetPredictandDatabase(const wxString& val);

    int GetLeadTimeNb() const {
        return (int)m_leadTimeDaysVect.size();
    }

    void SetLeadTimeDaysVector(vd val);

    void SetLeadTimeHoursVector(vd val);

    vd GetLeadTimeDaysVector() const {
        return m_leadTimeDaysVect;
    }

    void SetAnalogsNumberLeadTimeVector(int iStep, vi val);

    vi GetAnalogsNumberLeadTimeVector(int iStep) const {
        return m_stepsForecast[iStep].analogsNumberLeadTime;
    }

    int GetAnalogsNumberLeadTime(int iStep, int iLead) const {
        wxASSERT((int)m_stepsForecast[iStep].analogsNumberLeadTime.size() > iLead);
        return m_stepsForecast[iStep].analogsNumberLeadTime[iLead];
    }

    wxString GetPredictorArchiveDatasetId(int iStep, int iPtor) const {
        return m_stepsForecast[iStep].predictors[iPtor].archiveDatasetId;
    }

    void SetPredictorArchiveDatasetId(int iStep, int iPtor, const wxString& val);

    wxString GetPredictorArchiveDataId(int iStep, int iPtor) const {
        return m_stepsForecast[iStep].predictors[iPtor].archiveDataId;
    }

    void SetPredictorArchiveDataId(int iStep, int iPtor, const wxString& val);

    wxString GetPredictorRealtimeDatasetId(int iStep, int iPtor) const {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeDatasetId;
    }

    void SetPredictorRealtimeDatasetId(int iStep, int iPtor, const wxString& val);

    wxString GetPredictorRealtimeDataId(int iStep, int iPtor) const {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeDataId;
    }

    void SetPredictorRealtimeDataId(int iStep, int iPtor, const wxString& val);

    int GetPreprocessSize(int iStep, int iPtor) const override {
        return (int)m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.size();
    }

    void SetRealtimeStandardize(int iStep, int iPtor, const bool& val) {
        m_stepsForecast[iStep].predictors[iPtor].realtimeStandardize = val;
    }

    bool GetRealtimeStandardize(int iStep, int iPtor) {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeStandardize;
    }

    void SetRealtimeStandardizeMean(int iStep, int iPtor, const double& val) {
        m_stepsForecast[iStep].predictors[iPtor].realtimeStandardizeMean = val;
    }

    double GetRealtimeStandardizeMean(int iStep, int iPtor) {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeStandardizeMean;
    }

    void SetRealtimeStandardizeSd(int iStep, int iPtor, const double& val) {
        m_stepsForecast[iStep].predictors[iPtor].realtimeStandardizeSd = val;
    }

    double GetRealtimeStandardizeSd(int iStep, int iPtor) {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeStandardizeSd;
    }

    void SetArchiveStandardize(int iStep, int iPtor, const bool& val) {
        m_stepsForecast[iStep].predictors[iPtor].archiveStandardize = val;
    }

    bool GetArchiveStandardize(int iStep, int iPtor) {
        return m_stepsForecast[iStep].predictors[iPtor].archiveStandardize;
    }

    void SetArchiveStandardizeMean(int iStep, int iPtor, const double& val) {
        m_stepsForecast[iStep].predictors[iPtor].archiveStandardizeMean = val;
    }

    double GetArchiveStandardizeMean(int iStep, int iPtor) {
        return m_stepsForecast[iStep].predictors[iPtor].archiveStandardizeMean;
    }

    void SetArchiveStandardizeSd(int iStep, int iPtor, const double& val) {
        m_stepsForecast[iStep].predictors[iPtor].archiveStandardizeSd = val;
    }

    double GetArchiveStandardizeSd(int iStep, int iPtor) {
        return m_stepsForecast[iStep].predictors[iPtor].archiveStandardizeSd;
    }

    wxString GetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre, const wxString& val);

    wxString GetPreprocessArchiveDataId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessArchiveDataId(int iStep, int iPtor, int iPre, const wxString& val);

    wxString GetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre, const wxString& val);

    wxString GetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre, const wxString& val);

    int GetPredictorArchiveMembersNb(int iStep, int iPtor) const {
        return m_stepsForecast[iStep].predictors[iPtor].archiveMembersNb;
    }

    int GetPredictorRealtimeMembersNb(int iStep, int iPtor) const {
        return m_stepsForecast[iStep].predictors[iPtor].realtimeMembersNb;
    }

    int GetPreprocessArchiveMembersNb(int iStep, int iPtor, int iPre) const {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveMembersNb;
    }

    int GetPreprocessRealtimeMembersNb(int iStep, int iPtor, int iPre) const {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeMembersNb;
    }

  protected:
  private:
    vd m_leadTimeDaysVect;
    VectorParamsStepForecast m_stepsForecast;
    wxString m_predictandDatabase;

    bool ParseDescription(asFileParametersForecast& fileParams, const wxXmlNode* nodeProcess);

    bool ParseTimeProperties(asFileParametersForecast& fileParams, const wxXmlNode* nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersForecast& fileParams, int iStep, const wxXmlNode* nodeProcess);

    bool ParsePreprocessedPredictors(asFileParametersForecast& fileParams, int iStep, int iPtor,
                                     const wxXmlNode* nodeParam);

    bool ParseAnalogValuesParams(asFileParametersForecast& fileParams, const wxXmlNode* nodeProcess);
};

#endif
