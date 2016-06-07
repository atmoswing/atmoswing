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

#ifndef ASDATAPREDICTORARCHIVE_H
#define ASDATAPREDICTORARCHIVE_H

#include <asIncludes.h>
#include <asDataPredictor.h>

class asDataPredictorArchive
        : public asDataPredictor
{
public:
    asDataPredictorArchive(const wxString &dataId);

    virtual ~asDataPredictorArchive();

    static asDataPredictorArchive *GetInstance(const wxString &datasetId, const wxString &dataId,
                                               const wxString &directory = wxEmptyString);

    virtual bool Init();

    bool ClipToArea(asGeoAreaCompositeGrid *desiredArea);

    double GetOriginalProviderStart() const
    {
        return m_originalProviderStart;
    }

    void SetFileNamePattern(const wxString &val)
    {
        m_fileNamePattern = val;
    }

protected:
    double m_originalProviderStart;
    double m_originalProviderEnd;
    wxString m_fileNamePattern;

    virtual bool CheckTimeArray(asTimeArray &timeArray) const;

private:

};

#endif // ASDATAPREDICTORARCHIVE_H
