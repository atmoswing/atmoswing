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

    VectorInt GetFileParamInt(asFileParametersForecast &fileParams, const wxString &tag);

    VectorFloat GetFileParamFloat(asFileParametersForecast &fileParams, const wxString &tag);

    VectorDouble GetFileParamDouble(asFileParametersForecast &fileParams, const wxString &tag);

    VectorString GetFileParamString(asFileParametersForecast &fileParams, const wxString &tag);

    bool LoadFromFile(const wxString &filePath);

    void InitValues();

    int GetLeadTimeNb()
    {
        return m_LeadTimeDaysVect.size();
    }

    void SetLeadTimeDaysVector(VectorInt val)
    {
        m_LeadTimeDaysVect = val;
    }

    VectorInt GetLeadTimeDaysVector()
    {
        return m_LeadTimeDaysVect;
    }

    int GetLeadTimeDays(int i_leadtime)
    {
        return m_LeadTimeDaysVect[i_leadtime];
    }

    int GetLeadTimeHours(int i_leadtime)
    {
        return m_LeadTimeDaysVect[i_leadtime]*24.0;
    }

    void SetAnalogsNumberLeadTimeVector(int i_step, VectorInt val)
    {
        m_StepsForecast[i_step].AnalogsNumberLeadTime = val;
    }

    VectorInt GetAnalogsNumberLeadTimeVector(int i_step)
    {
        return m_StepsForecast[i_step].AnalogsNumberLeadTime;
    }

    int GetAnalogsNumberLeadTime(int i_step, int i_leadtime)
    {
        return m_StepsForecast[i_step].AnalogsNumberLeadTime[i_leadtime];
    }

    int GetAnalogsNumberLeadTimeLastStep(int i_leadtime)
    {
        return m_StepsForecast[m_StepsForecast.size()-1].AnalogsNumberLeadTime[i_leadtime];
    }

    wxString GetPredictorArchiveDatasetId(int i_step, int i_predictor)
    {
        return m_StepsForecast[i_step].Predictors[i_predictor].ArchiveDatasetId;
    }

    void SetPredictorArchiveDatasetId(int i_step, int i_predictor, const wxString& val)
    {
        m_StepsForecast[i_step].Predictors[i_predictor].ArchiveDatasetId = val;
    }

    wxString GetPredictorArchiveDataId(int i_step, int i_predictor)
    {
        return m_StepsForecast[i_step].Predictors[i_predictor].ArchiveDataId;
    }

    void SetPredictorArchiveDataId(int i_step, int i_predictor, const wxString& val)
    {
        m_StepsForecast[i_step].Predictors[i_predictor].ArchiveDataId = val;
    }

    wxString GetPredictorRealtimeDatasetId(int i_step, int i_predictor)
    {
        return m_StepsForecast[i_step].Predictors[i_predictor].RealtimeDatasetId;
    }

    void SetPredictorRealtimeDatasetId(int i_step, int i_predictor, const wxString& val)
    {
        m_StepsForecast[i_step].Predictors[i_predictor].RealtimeDatasetId = val;
    }

    wxString GetPredictorRealtimeDataId(int i_step, int i_predictor)
    {
        return m_StepsForecast[i_step].Predictors[i_predictor].RealtimeDataId;
    }

    void SetPredictorRealtimeDataId(int i_step, int i_predictor, const wxString& val)
    {
        m_StepsForecast[i_step].Predictors[i_predictor].RealtimeDataId = val;
    }

    int GetPreprocessSize(int i_step, int i_predictor)
    {
        return m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDatasetIds.size();
    }

    wxString GetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDatasetIds.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDatasetIds[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessArchiveDatasetIds in the parameters object."));
            return wxEmptyString;
        }
    }

    void SetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDatasetIds.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDatasetIds[i_dataset] = val;
        }
        else
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDatasetIds.push_back(val);
        }
    }

    wxString GetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDataIds.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDataIds[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessArchiveDatasetIds in the parameters object."));
            return wxEmptyString;
        }
    }

    void SetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDataIds.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDataIds[i_dataset] = val;
        }
        else
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessArchiveDataIds.push_back(val);
        }
    }

    wxString GetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDatasetIds.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDatasetIds[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessRealtimeDatasetIds in the parameters object."));
            return wxEmptyString;
        }
    }

    void SetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDatasetIds.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDatasetIds[i_dataset] = val;
        }
        else
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDatasetIds.push_back(val);
        }
    }

    wxString GetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDataIds.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDataIds[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessRealtimeDatasetIds in the parameters object."));
            return wxEmptyString;
        }
    }

    void SetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if(m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDataIds.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDataIds[i_dataset] = val;
        }
        else
        {
            m_StepsForecast[i_step].Predictors[i_predictor].PreprocessRealtimeDataIds.push_back(val);
        }
    }


protected:

private:
    VectorInt m_LeadTimeDaysVect;
    VectorParamsStepForecast m_StepsForecast;
};

#endif // ASPARAMETERSFORECAST_H
