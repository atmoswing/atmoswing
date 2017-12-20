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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#ifndef ASPARAMETERSDOWNSCALING_H
#define ASPARAMETERSDOWNSCALING_H

#include "asIncludes.h"
#include <asParameters.h>

class asFileParametersDownscaling;


class asParametersDownscaling
        : public asParameters
{
public:
    typedef struct
    {
        wxString datasetId;
        wxString dataId;
        int membersNb;
        vwxs preprocessDatasetIds;
        vwxs preprocessDataIds;
        int preprocessMembersNb;
    } ParamsPredictorScenario;

    typedef std::vector<ParamsPredictorScenario> VectorParamsPredictorsScenario;

    typedef struct
    {
        VectorParamsPredictorsScenario predictors;
    } ParamsStepScenario;

    typedef std::vector<ParamsStepScenario> VectorParamsStepScenario;

    asParametersDownscaling();

    virtual ~asParametersDownscaling();

    void AddStep();

    void AddPredictorScenario(ParamsStepScenario &step);

    bool LoadFromFile(const wxString &filePath);

    bool InputsOK() const;

    bool FixTimeLimits();

    void InitValues();

    vvi GetPredictandStationIdsVector() const
    {
        return m_predictandStationIdsVect;
    }

    bool SetPredictandStationIdsVector(vvi val);

    wxString GetPredictorScenarioDatasetId(int iStep, int iPtor) const
    {
        return m_stepsScenarion[iStep].predictors[iPtor].datasetId;
    }

    bool SetPredictorScenarioDatasetId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorScenarioDataId(int iStep, int iPtor) const
    {
        return m_stepsScenarion[iStep].predictors[iPtor].dataId;
    }

    bool SetPredictorScenarioDataId(int iStep, int iPtor, const wxString &val);

    wxString GetPreprocessScenarioDatasetId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessScenarioDatasetId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessScenarioDataId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessScenarioDataId(int iStep, int iPtor, int iPre, const wxString &val);

    void SetPredictorScenarioMembersNb(int iStep, int iPtor, int val)
    {
        m_stepsScenarion[iStep].predictors[iPtor].membersNb = val;
    }

    int GetPredictorScenarioMembersNb(int iStep, int iPtor) const
    {
        return m_stepsScenarion[iStep].predictors[iPtor].membersNb;
    }

    void SetPreprocessScenarioMembersNb(int iStep, int iPtor, int iPre, int val)
    {
        m_stepsScenarion[iStep].predictors[iPtor].preprocessMembersNb = val;
    }

    int GetPreprocessScenarioMembersNb(int iStep, int iPtor, int iPre) const
    {
        return m_stepsScenarion[iStep].predictors[iPtor].preprocessMembersNb;
    }

    bool SetDownscalingYearStart(int val)
    {
        m_downscalingStart = asTime::GetMJD(val, 1, 1);
        return true;
    }

    bool SetDownscalingYearEnd(int val)
    {
        m_downscalingEnd = asTime::GetMJD(val, 12, 31);
        return true;
    }

    double GetDownscalingStart() const
    {
        return m_downscalingStart;
    }

    bool SetDownscalingStart(wxString val)
    {
        m_downscalingStart = asTime::GetTimeFromString(val);
        return true;
    }

    double GetDownscalingEnd() const
    {
        return m_downscalingEnd;
    }

    bool SetDownscalingEnd(wxString val)
    {
        m_downscalingEnd = asTime::GetTimeFromString(val);
        return true;
    }

protected:
    double m_downscalingStart;
    double m_downscalingEnd;

private:
    vvi m_predictandStationIdsVect;
    VectorParamsStepScenario m_stepsScenarion;

    bool ParseDescription(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersDownscaling &fileParams, int iStep, const wxXmlNode *nodeProcess);

    bool ParsePreprocessedPredictors(asFileParametersDownscaling &fileParams, int iStep, int iPtor,
                                     const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

};

#endif // ASPARAMETERSDOWNSCALING_H
