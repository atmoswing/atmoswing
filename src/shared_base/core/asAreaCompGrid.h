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

#ifndef AS_AREA_COMPOSITE_GRID_H
#define AS_AREA_COMPOSITE_GRID_H

#include "asAreaComp.h"
#include "asIncludes.h"

class asParameters;

class asAreaCompGrid : public asAreaComp {
   public:
    static asAreaCompGrid *GetInstance(const asParameters *params, int iStep, int iPtor);

    static asAreaCompGrid *GetInstance(const wxString &type, double xMin, int xPtsNb, double xStep, double yMin,
                                       int yPtsNb, double yStep, int flatAllowed = asFLAT_FORBIDDEN,
                                       bool isLatLon = true);

    static asAreaCompGrid *GetInstance(double xMin, int xPtsNb, double xStep, double yMin, int yPtsNb, double yStep,
                                       int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asAreaCompGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                   int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asAreaCompGrid(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed = asFLAT_FORBIDDEN,
                   bool isLatLon = true);

    asAreaCompGrid();

    virtual bool InitializeAxes(const a1d &lons, const a1d &lats, bool strideAllowed = true, bool getLarger = false);

    virtual bool GridsOverlay(asAreaCompGrid *otherArea) const = 0;

    void CorrectCornersWithAxes();

    a1d GetXaxis();

    a1d GetYaxis();

    a1d GetXaxisComposite(int compositeNb);

    a1d GetYaxisComposite(int compositeNb);

    int GetXaxisCompositePtsnb(int compositeNb);

    int GetYaxisCompositePtsnb(int compositeNb);

    double GetXaxisCompositeStart(int compositeNb) const;

    double GetYaxisCompositeStart(int compositeNb) const;

    double GetXaxisCompositeEnd(int compositeNb) const;

    double GetYaxisCompositeEnd(int compositeNb) const;

    int GetXptsNb();

    int GetYptsNb();

    double GetXmin() const;

    double GetXmax() const;

    double GetYmin() const;

    double GetYmax() const;

    virtual double GetXstep() const = 0;

    virtual double GetYstep() const = 0;

    bool IsRegular() const {
        return m_isRegular;
    }

    void AllowResizeFromData() {
        m_allowResizeFromData = true;
    }

   protected:
    bool m_isRegular;
    bool m_isInitialized;
    bool m_allowResizeFromData;
    va1d m_compositeXaxes;
    va1d m_compositeYaxes;
    int m_xPtsNb;
    int m_yPtsNb;

   private:
    bool CreateCompositeAxes(const a1d &lons, const a1d &lats, bool getLarger = false);

    bool AreaDefinedByPointsNb(const a1d &lons, const a1d &lats);

    bool HandleAreaDefinedByPointsNb(const a1d &lons, const a1d &lats);

    void HandleNegativeLongitudes(const a1d &lons);

    void HandleLongitudesAbove360(const a1d &lons);

    void HandleLongitudesSplitAt180(const a1d &lons);

    void HandleMissing360(const a1d &lons);

    void HandleMissing180(const a1d &lons);
};

#endif
