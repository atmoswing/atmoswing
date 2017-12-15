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

    asGeoAreaCompositeGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                           float level = asNONE, float height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(double xMin, double xWidth, double yMin, double yWidth, float level = asNONE,
                           float height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(float level = asNONE, float height = asNONE);

    static asGeoAreaCompositeGrid *GetInstance(const wxString &type, double xMin, int xPtsNb, double xStep, double yMin,
                                               int yPtsNb, double yStep, float level = asNONE, float height = asNONE,
                                               int flatAllowed = asFLAT_FORBIDDEN);

    static a1d GetXaxis(const wxString &type, double xMin, double xMax, double xStep = 0);

    static a1d GetYaxis(const wxString &type, double yMin, double yMax, double yStep = 0);

    virtual bool GridsOverlay(asGeoAreaCompositeGrid *otherarea) const = 0;

    void SetLastRowAsNewComposite();

    void RemoveLastRowOnComposite(int i);

    virtual double GetXstep() const = 0;

    virtual double GetYstep() const = 0;

    virtual a1d GetXaxisComposite(int compositeNb) = 0;

    virtual a1d GetYaxisComposite(int compositeNb) = 0;

    virtual int GetXaxisCompositePtsnb(int compositeNb) = 0;

    virtual int GetYaxisCompositePtsnb(int compositeNb) = 0;

    virtual double GetXaxisCompositeWidth(int compositeNb) const = 0;

    virtual double GetYaxisCompositeWidth(int compositeNb) const = 0;

    virtual double GetXaxisCompositeStart(int compositeNb) const = 0;

    virtual double GetYaxisCompositeStart(int compositeNb) const = 0;

    virtual double GetXaxisCompositeEnd(int compositeNb) const = 0;

    virtual double GetYaxisCompositeEnd(int compositeNb) const = 0;

    int GetXaxisPtsnb();

    int GetYaxisPtsnb();

    double GetXaxisWidth() const;

    double GetYaxisWidth() const;

    a1d GetXaxis();

    a1d GetYaxis();

protected:

private:

};

#endif // asGeoAreaCompositeGrid_H
