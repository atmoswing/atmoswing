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
        wxString archiveDatasetId;
        wxString archiveDataId;
        int archiveMembersNb;
        wxString modelSimDatasetId;
        wxString modelSimDataId;
        int modelSimMembersNb;
        vwxs preprocessArchiveDatasetIds;
        vwxs preprocessArchiveDataIds;
        int preprocessArchiveMembersNb;
        vwxs preprocessModelSimDatasetIds;
        vwxs preprocessModelSimDataIds;
        int preprocessModelSimMembersNb;
    } ParamsPredictorModelSim;

    typedef std::vector<ParamsPredictorModelSim> VectorParamsPredictorsModelSim;

    typedef struct
    {
        VectorParamsPredictorsModelSim predictors;
    } ParamsStepModelSim;

    typedef std::vector<ParamsStepModelSim> VectorParamsStepModelSim;

    asParametersDownscaling();

    virtual ~asParametersDownscaling();

    void AddStep();

    void AddPredictorModelSim(ParamsStepModelSim &step);

    bool LoadFromFile(const wxString &filePath);

    bool InputsOK() const;

    bool FixTimeLimits();

    void InitValues();

    vvi GetPredictandStationIdsVector() const
    {
        return m_predictandStationIdsVect;
    }

    bool SetPredictandStationIdsVector(vvi val);

    wxString GetPredictorArchiveDatasetId(int iStep, int iPtor) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].archiveDatasetId;
    }

    bool SetPredictorArchiveDatasetId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorArchiveDataId(int iStep, int iPtor) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].archiveDataId;
    }

    bool SetPredictorArchiveDataId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorModelSimDatasetId(int iStep, int iPtor) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].modelSimDatasetId;
    }

    bool SetPredictorModelSimDatasetId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorModelSimDataId(int iStep, int iPtor) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].modelSimDataId;
    }

    bool SetPredictorModelSimDataId(int iStep, int iPtor, const wxString &val);

    int GetPreprocessSize(int iStep, int iPtor) const
    {
        return (int) m_stepsModelSim[iStep].predictors[iPtor].preprocessArchiveDatasetIds.size();
    }

    wxString GetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessArchiveDataId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessArchiveDataId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessModelSimDatasetId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessModelSimDatasetId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessModelSimDataId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessModelSimDataId(int iStep, int iPtor, int iPre, const wxString &val);


    void SetPredictorArchiveMembersNb(int iStep, int iPtor, int val)
    {
        m_stepsModelSim[iStep].predictors[iPtor].archiveMembersNb = val;
    }

    int GetPredictorArchiveMembersNb(int iStep, int iPtor) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].archiveMembersNb;
    }

    void SetPredictorModelSimMembersNb(int iStep, int iPtor, int val)
    {
        m_stepsModelSim[iStep].predictors[iPtor].modelSimMembersNb = val;
    }

    int GetPredictorModelSimMembersNb(int iStep, int iPtor) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].modelSimMembersNb;
    }

    void SetPreprocessArchiveMembersNb(int iStep, int iPtor, int iPre, int val)
    {
        m_stepsModelSim[iStep].predictors[iPtor].preprocessArchiveMembersNb = val;
    }

    int GetPreprocessArchiveMembersNb(int iStep, int iPtor, int iPre) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].preprocessArchiveMembersNb;
    }

    void SetPreprocessModelSimMembersNb(int iStep, int iPtor, int iPre, int val)
    {
        m_stepsModelSim[iStep].predictors[iPtor].preprocessModelSimMembersNb = val;
    }

    int GetPreprocessModelSimMembersNb(int iStep, int iPtor, int iPre) const
    {
        return m_stepsModelSim[iStep].predictors[iPtor].preprocessModelSimMembersNb;
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
    VectorParamsStepModelSim m_stepsModelSim;

    bool ParseDescription(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersDownscaling &fileParams, int iStep, const wxXmlNode *nodeProcess);

    bool ParsePreprocessedPredictors(asFileParametersDownscaling &fileParams, int iStep, int iPtor,
                                     const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

};

#endif // ASPARAMETERSDOWNSCALING_H
