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

#ifndef ASPREDICTORARCHNCEPREANALYSIS1LTHE_H
#define ASPREDICTORARCHNCEPREANALYSIS1LTHE_H

#include <asIncludes.h>
#include <asPredictorArchNcepReanalysis1Subset.h>

class asGeoArea;

class asPredictorArchNcepReanalysis1Lthe
        : public asPredictorArchNcepReanalysis1Subset
{
public:
    asPredictorArchNcepReanalysis1Lthe(const wxString &dataId);

    virtual ~asPredictorArchNcepReanalysis1Lthe();

    virtual bool Init();

protected:
    virtual vwxs GetListOfFiles(asTimeArray &timeArray) const;

    virtual bool ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                 vvva2f &compositeData);

    virtual double ConvertToMjd(double timeValue, double refValue = NaNd) const;

private:

};

#endif // ASPREDICTORARCHNCEPREANALYSIS1LTHE_H