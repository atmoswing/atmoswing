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
    int GetForecastsNb();
    void ClearForecasts();
    void AddForecast();;

    wxString GetFilePath()
    {
        return m_filePath;
    }
    
    void SetFilePath(const wxString &path)
    {
        m_filePath = path;
    }

    bool HasChanged()
    {
        return m_hasChanged;
    }

    bool HasExports();

    bool ExportSyntheticXml()
    {
        return m_exportSyntheticXml;
    }

    void SetExportSyntheticXml(bool val)
    {
        m_exportSyntheticXml = val;
    }

    void SetHasChanged(bool val)
    {
        m_hasChanged = val;
    }

    wxString GetForecastsOutputDirectory()
    {
        return m_forecastsOutputDirectory;
    }
    
    void SetForecastsOutputDirectory(const wxString &val)
    {
        m_forecastsOutputDirectory = val;
    }

    wxString GetExportsOutputDirectory()
    {
        return m_exportsOutputDirectory;
    }

    void SetExportsOutputDirectory(const wxString &val)
    {
        m_exportsOutputDirectory = val;
    }

    wxString GetParametersFileDirectory()
    {
        return m_parametersFileDirectory;
    }
    
    void SetParametersFileDirectory(const wxString &val)
    {
        m_parametersFileDirectory = val;
    }

    wxString GetPredictorsArchiveDirectory()
    {
        return m_predictorsArchiveDirectory;
    }
    
    void SetPredictorsArchiveDirectory(const wxString &val)
    {
        m_predictorsArchiveDirectory = val;
    }

    wxString GetPredictorsRealtimeDirectory()
    {
        return m_predictorsRealtimeDirectory;
    }
    
    void SetPredictorsRealtimeDirectory(const wxString &val)
    {
        m_predictorsRealtimeDirectory = val;
    }

    wxString GetPredictandDBDirectory()
    {
        return m_predictandDBDirectory;
    }
    
    void SetPredictandDBDirectory(const wxString &val)
    {
        m_predictandDBDirectory = val;
    }

    wxString GetForecastFileName(int i)
    {
        wxASSERT(m_forecastFileNames.size()>i);
        return m_forecastFileNames[i];
    }

    void SetForecastFileName(int i, const wxString &val)
    {
        wxASSERT(m_forecastFileNames.size()>i);
        m_forecastFileNames[i] = val;
    }

protected:
private:
    bool m_hasChanged;
    bool m_exportSyntheticXml;
    wxString m_filePath;
    wxString m_forecastsOutputDirectory;
    wxString m_exportsOutputDirectory;
    wxString m_parametersFileDirectory;
    wxString m_predictorsArchiveDirectory;
    wxString m_predictorsRealtimeDirectory;
    wxString m_predictandDBDirectory;
    VectorString m_forecastFileNames;

};

#endif // ASBATCHFORECASTS_H
