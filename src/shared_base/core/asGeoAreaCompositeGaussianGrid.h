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

#ifndef asGeoAreaCompositeGaussianGrid_H
#define asGeoAreaCompositeGaussianGrid_H

#include <asIncludes.h>
#include <asGeoAreaCompositeGrid.h>
#include <asGeoAreaGaussianGrid.h>

class asGeoAreaCompositeGaussianGrid
        : public asGeoAreaCompositeGrid
{
public:
    asGeoAreaCompositeGaussianGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR,
                                   double Xstep, double Ystep,
                                   asGeoAreaGaussianGrid::GaussianGridType type = asGeoAreaGaussianGrid::T62,
                                   float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGaussianGrid(double Xmin, int Xptsnb, double Ymin, int Yptsnb,
                                   asGeoAreaGaussianGrid::GaussianGridType type = asGeoAreaGaussianGrid::T62,
                                   float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    virtual ~asGeoAreaCompositeGaussianGrid();

    bool GridsOverlay(asGeoAreaCompositeGrid *otherarea);

    asGeoAreaGaussianGrid::GaussianGridType GetGaussianGridType()
    {
        return m_gaussianGridType;
    }

    double GetXstep()
    {
        return 0.0;
    }

    double GetYstep()
    {
        return 0.0;
    }

    Array1DDouble GetXaxisComposite(int compositeNb);

    Array1DDouble GetYaxisComposite(int compositeNb);

    int GetXaxisCompositePtsnb(int compositeNb);

    int GetYaxisCompositePtsnb(int compositeNb);

    double GetXaxisCompositeWidth(int compositeNb);

    double GetYaxisCompositeWidth(int compositeNb);

    double GetXaxisCompositeStart(int compositeNb);

    double GetYaxisCompositeStart(int compositeNb);

    double GetXaxisCompositeEnd(int compositeNb);

    double GetYaxisCompositeEnd(int compositeNb);

protected:

private:
    asGeoAreaGaussianGrid::GaussianGridType m_gaussianGridType;
    Array1DDouble m_fullAxisX;
    Array1DDouble m_fullAxisY;

    bool IsOnGrid(const Coo &point);

    bool IsOnGrid(double Xcoord, double Ycoord);
};

#endif // asGeoAreaCompositeGaussianGrid_H
