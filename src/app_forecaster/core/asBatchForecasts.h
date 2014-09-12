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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#ifndef ASBATCHFORECASTS_H
#define ASBATCHFORECASTS_H

#include <asIncludes.h>
#include <asFileBatchForecasts.h>

class asBatchForecasts : public wxObject
{
public:
    /** Default constructor */
    asBatchForecasts();

    /** Default destructor */
    virtual ~asBatchForecasts();

    bool Load(const wxString &filePath);
    bool Save();
    int GetModelsNb();
    void ClearModels();
    void AddModel();;

    wxString GetFilePath()
    {
        return m_FilePath;
    }
    
    void SetFilePath(const wxString &path)
    {
        m_FilePath = path;
    }

    bool HasChanged()
    {
        return m_HasChanged;
    }

    void SetHasChanged(bool val)
    {
        m_HasChanged = val;
    }

    wxString GetForecastsOutputDirectory()
    {
        return m_ForecastsOutputDirectory;
    }
    
    void SetForecastsOutputDirectory(const wxString &val)
    {
        m_ForecastsOutputDirectory = val;
    }

    wxString GetParametersFileDirectory()
    {
        return m_ParametersFileDirectory;
    }
    
    void SetParametersFileDirectory(const wxString &val)
    {
        m_ParametersFileDirectory = val;
    }

    wxString GetPredictorsArchiveDirectory()
    {
        return m_PredictorsArchiveDirectory;
    }
    
    void SetPredictorsArchiveDirectory(const wxString &val)
    {
        m_PredictorsArchiveDirectory = val;
    }

    wxString GetPredictorsRealtimeDirectory()
    {
        return m_PredictorsRealtimeDirectory;
    }
    
    void SetPredictorsRealtimeDirectory(const wxString &val)
    {
        m_PredictorsRealtimeDirectory = val;
    }

    wxString GetPredictandDBDirectory()
    {
        return m_PredictandDBDirectory;
    }
    
    void SetPredictandDBDirectory(const wxString &val)
    {
        m_PredictandDBDirectory = val;
    }

    wxString GetModelName(int i)
    {
        wxASSERT(m_ModelNames.size()>i);
        return m_ModelNames[i];
    }

    void SetModelName(int i, const wxString &val)
    {
        wxASSERT(m_ModelNames.size()>i);
        m_ModelNames[i] = val;
    }

    wxString GetModelDescription(int i)
    {
        wxASSERT(m_ModelDescriptions.size()>i);
        return m_ModelDescriptions[i];
    }

    void SetModelDescription(int i, const wxString &val)
    {
        wxASSERT(m_ModelDescriptions.size()>i);
        m_ModelDescriptions[i] = val;
    }

    wxString GetModelFileName(int i)
    {
        wxASSERT(m_ModelFileNames.size()>i);
        return m_ModelFileNames[i];
    }

    void SetModelFileName(int i, const wxString &val)
    {
        wxASSERT(m_ModelFileNames.size()>i);
        m_ModelFileNames[i] = val;
    }

    wxString GetModelPredictandDB(int i)
    {
        wxASSERT(m_ModelPredictandDBs.size()>i);
        return m_ModelPredictandDBs[i];
    }

    void SetModelPredictandDB(int i, const wxString &val)
    {
        wxASSERT(m_ModelPredictandDBs.size()>i);
        m_ModelPredictandDBs[i] = val;
    }

protected:
private:
    bool m_HasChanged;
    wxString m_FilePath;
    wxString m_ForecastsOutputDirectory;
    wxString m_ParametersFileDirectory;
    wxString m_PredictorsArchiveDirectory;
    wxString m_PredictorsRealtimeDirectory;
    wxString m_PredictandDBDirectory;
    VectorString m_ModelNames;
    VectorString m_ModelDescriptions;
    VectorString m_ModelFileNames;
    VectorString m_ModelPredictandDBs;

};

#endif // ASBATCHFORECASTS_H
