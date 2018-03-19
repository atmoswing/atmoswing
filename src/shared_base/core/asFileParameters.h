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

#ifndef ASFILEPARAMETERS_H
#define ASFILEPARAMETERS_H

#include <asIncludes.h>
#include <asFileXml.h>

class asFileParameters
        : public asFileXml
{
public:
    asFileParameters(const wxString &fileName, const FileMode &fileMode);

    virtual ~asFileParameters();

    virtual bool EditRootElement();

    virtual bool CheckRootElement() const;

    static vi BuildVectorInt(int min, int max, int step);

    static vi BuildVectorInt(wxString str);

    static vf BuildVectorFloat(float min, float max, float step);

    static vf BuildVectorFloat(wxString str);

    static vd BuildVectorDouble(double min, double max, double step);

    static vd BuildVectorDouble(wxString str);

    static vwxs BuildVectorString(wxString str);

    static vi GetVectorInt(wxXmlNode *node);

    static vf GetVectorFloat(wxXmlNode *node);

    static vd GetVectorDouble(wxXmlNode *node);

    static vwxs GetVectorString(wxXmlNode *node);

    static vvi GetStationIdsVector(wxXmlNode *node);

    static vi GetStationIds(wxString stationIdsString);

protected:

private:

};

#endif // ASFILEPARAMETERS_H
