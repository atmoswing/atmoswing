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

#ifndef asGeoAreaGaussianGrid_H
#define asGeoAreaGaussianGrid_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaGaussianGrid
        : public asGeoArea
{
public:
    asGeoAreaGaussianGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR,
                          asGeo::GridType type, float Level = asNONE, float Height = asNONE,
                          int flatAllowed = asFLAT_ALLOWED);

    asGeoAreaGaussianGrid(double Xmin, int Xptsnb, double Ymin, int Yptsnb, asGeo::GridType type,
                          float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    virtual ~asGeoAreaGaussianGrid();

    static void BuildLonAxis(Array1DDouble &axis, const asGeo::GridType &type);

    static void BuildLatAxis(Array1DDouble &axis, const asGeo::GridType &type);

    int GetXaxisPtsnb() const;

    int GetYaxisPtsnb() const;

    Array1DDouble GetXaxis();

    Array1DDouble GetYaxis();

protected:

private:
    Array1DDouble m_fullAxisX;
    Array1DDouble m_fullAxisY;

    bool IsOnGrid(const Coo &point) const;

    bool IsOnGrid(double Xcoord, double Ycoord) const;
};

#endif // asGeoAreaGaussianGrid_H
