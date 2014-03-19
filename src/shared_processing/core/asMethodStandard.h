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
 
#ifndef ASMETHODSTANDARD_H
#define ASMETHODSTANDARD_H

#include <asIncludes.h>
#include <asDataPredictand.h>


class asMethodStandard: public wxObject
{
public:
    /** Default constructor */
    asMethodStandard();
    /** Default destructor */
    virtual ~asMethodStandard();

    virtual bool Manager();

    bool LoadPredictandDB(const wxString &predictandDBFilePath);

    void Cancel();

    /** Access m_ParamsFilePath
     * \return The current value of m_ParamsFilePath
     */
    wxString GetParamsFilePath()
    {
        return m_ParamsFilePath;
    }

    /** Set m_ParamsFilePath
     * \param val New value to set
     */
    void SetParamsFilePath(const wxString &val)
    {
        m_ParamsFilePath = val;
    }

    /** Access m_PredictandDBFilePath
     * \return The current value of m_PredictandDBFilePath
     */
    wxString GetPredictandDBFilePath()
    {
        return m_PredictandDBFilePath;
    }

    /** Set m_PredictandDBFilePath
     * \param val New value to set
     */
    void SetPredictandDBFilePath(const wxString &val)
    {
        m_PredictandDBFilePath = val;
    }

    /** Access m_PredictandDB
     * \return The current m_PredictandDB pointer
     */
    asDataPredictand* GetPredictandDB()
    {
        return m_PredictandDB;
    }

    /** Set m_PredictandDB
     * \param pDB Pointer to the DB
     */
    void SetPredictandDB(asDataPredictand* pDB)
    {
        m_PredictandDB = pDB;
    }

    /** Access m_PredictorDataDir
     * \return The current value of m_PredictorDataDir
     */
    wxString GetPredictorDataDir()
    {
        return m_PredictorDataDir;
    }

    /** Set m_PredictorDataDir
     * \param val New value to set
     */
    void SetPredictorDataDir(const wxString &val)
    {
        m_PredictorDataDir = val;
    }

protected:
    bool m_Cancel;
    wxString m_ParamsFilePath;
    wxString m_PredictandDBFilePath;
    wxString m_PredictorDataDir;
    // TODO (Pascal#5#): Make it compatible for temperature predictand DB.
    asDataPredictand* m_PredictandDB;

private:
};

#endif // ASMETHODSTANDARD_H
