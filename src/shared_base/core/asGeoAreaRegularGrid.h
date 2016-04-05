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

#ifndef asGeoAreaRegularGrid_H
#define asGeoAreaRegularGrid_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaRegularGrid
        : public asGeoArea
{
public:
    asGeoAreaRegularGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR,
                         double Xstep, double Ystep, float Level = asNONE, float Height = asNONE,
                         int flatAllowed = asFLAT_ALLOWED);

    asGeoAreaRegularGrid(double Xmin, double Xwidth, double Xstep, double Ymin, double Ywidth, double Ystep,
                         float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    virtual ~asGeoAreaRegularGrid();

    double GetXstep() const
    {
        return m_xStep;
    }

    double GetYstep() const
    {
        return m_yStep;
    }

    int GetXaxisPtsnb() const;

    int GetYaxisPtsnb() const;

    Array1DDouble GetXaxis() const;

    Array1DDouble GetYaxis() const;

protected:

private:
    double m_xStep;
    double m_yStep;

    bool IsOnGrid(double step) const;

    bool IsOnGrid(double stepX, double stepY) const;
};

#endif // asGeoAreaRegularGrid_H
