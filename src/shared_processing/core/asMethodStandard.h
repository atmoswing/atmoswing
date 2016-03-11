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

    /** Access m_paramsFilePath
     * \return The current value of m_paramsFilePath
     */
    wxString GetParamsFilePath()
    {
        return m_paramsFilePath;
    }

    /** Set m_paramsFilePath
     * \param val New value to set
     */
    void SetParamsFilePath(const wxString &val)
    {
        m_paramsFilePath = val;
    }

    /** Access m_predictandDBFilePath
     * \return The current value of m_predictandDBFilePath
     */
    wxString GetPredictandDBFilePath()
    {
        return m_predictandDBFilePath;
    }

    /** Set m_predictandDBFilePath
     * \param val New value to set
     */
    void SetPredictandDBFilePath(const wxString &val)
    {
        m_predictandDBFilePath = val;
    }

    /** Access m_predictandDB
     * \return The current m_predictandDB pointer
     */
    asDataPredictand* GetPredictandDB()
    {
        return m_predictandDB;
    }

    /** Set m_predictandDB
     * \param pDB Pointer to the DB
     */
    void SetPredictandDB(asDataPredictand* pDB)
    {
        m_predictandDB = pDB;
    }

    /** Access m_predictorDataDir
     * \return The current value of m_predictorDataDir
     */
    wxString GetPredictorDataDir()
    {
        return m_predictorDataDir;
    }

    /** Set m_predictorDataDir
     * \param val New value to set
     */
    void SetPredictorDataDir(const wxString &val)
    {
        m_predictorDataDir = val;
    }

protected:
    bool m_cancel;
    wxString m_paramsFilePath;
    wxString m_predictandDBFilePath;
    wxString m_predictorDataDir;
    // TODO (Pascal#5#): Make it compatible for temperature predictand DB.
    asDataPredictand* m_predictandDB;

private:
};

#endif // ASMETHODSTANDARD_H
