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

#ifndef asGeoAreaCompositeGrid_H
#define asGeoAreaCompositeGrid_H

#include <asIncludes.h>
#include <asGeoAreaComposite.h>

class asGeoAreaCompositeGrid
        : public asGeoAreaComposite // Abstract class
{
public:
    enum GridType //!< Define available parameters sets (for the GUI)
    {
        Regular, GaussianT62
    };

    asGeoAreaCompositeGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR,
                           float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level = asNONE,
                           float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(float Level = asNONE, float Height = asNONE);

    static asGeoAreaCompositeGrid *GetInstance(const wxString &type, double Xmin, int Xptsnb, double Xstep, double Ymin,
                                               int Yptsnb, double Ystep, float Level = asNONE, float Height = asNONE,
                                               int flatAllowed = asFLAT_FORBIDDEN);

    virtual bool GridsOverlay(asGeoAreaCompositeGrid *otherarea) = 0;

    GridType GetGridType()
    {
        return m_gridType;
    }

    wxString GetGridTypeString()
    {
        switch (m_gridType) {
            case (Regular):
                return "Regular";
            case (GaussianT62):
                return "GaussianT62";
            default:
                return "Not found";
        }
    }

    virtual double GetXstep() = 0;

    virtual double GetYstep() = 0;

    virtual Array1DDouble GetXaxisComposite(int compositeNb) = 0;

    virtual Array1DDouble GetYaxisComposite(int compositeNb) = 0;

    virtual int GetXaxisCompositePtsnb(int compositeNb) = 0;

    virtual int GetYaxisCompositePtsnb(int compositeNb) = 0;

    virtual double GetXaxisCompositeWidth(int compositeNb) = 0;

    virtual double GetYaxisCompositeWidth(int compositeNb) = 0;

    virtual double GetXaxisCompositeStart(int compositeNb) = 0;

    virtual double GetYaxisCompositeStart(int compositeNb) = 0;

    virtual double GetXaxisCompositeEnd(int compositeNb) = 0;

    virtual double GetYaxisCompositeEnd(int compositeNb) = 0;

    int GetXaxisPtsnb();

    int GetYaxisPtsnb();

    double GetXaxisWidth();

    double GetYaxisWidth();

    Array1DDouble GetXaxis();

    Array1DDouble GetYaxis();

protected:
    GridType m_gridType;

private:

};

#endif // asGeoAreaCompositeGrid_H
