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
 
#ifndef ASDATAPREDICTORARCHIVE_H
#define ASDATAPREDICTORARCHIVE_H

#include <asIncludes.h>
#include <asDataPredictor.h>

class asDataPredictorArchive: public asDataPredictor
{
public:

    /** Default constructor
     * \param dataId The predictor data id
     */
    asDataPredictorArchive(const wxString &dataId);

    /** Default destructor */
    virtual ~asDataPredictorArchive();

    static asDataPredictorArchive* GetInstance(const wxString &datasetId, const wxString &dataId, const wxString &directory = wxEmptyString);
    
    virtual bool Init();

    bool ClipToArea(asGeoAreaCompositeGrid *desiredArea);
    
    /** Access m_OriginalProviderStart
     * \return The current value of m_OriginalProviderStart
     */
    int GetOriginalProviderStart()
    {
        return m_OriginalProviderStart;
    }

    void SetFileNamePattern(const wxString &val)
    {
        m_FileNamePattern = val;
    }

protected:
    double m_OriginalProviderStart;
    double m_OriginalProviderEnd;
    wxString m_SubFolder;
    wxString m_FileNamePattern;

    /** Method to check the time array compatibility with the data
     * \param timeArray The time array to check
     * \return True if compatible with the data
     */
    virtual bool CheckTimeArray(asTimeArray &timeArray);

private:

};

#endif // ASDATAPREDICTORARCHIVE_H
