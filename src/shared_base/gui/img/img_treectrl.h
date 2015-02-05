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
 * Portions Copyright 2015 Pascal Horton, Terr@num.
 */

#ifndef _TREECTRL_H_
#define _TREECTRL_H_ 1

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#    include <wx/wx.h>
#endif

#ifdef _MSC_VER
    #pragma warning( disable : 4125 ) // C4125: decimal digit terminates octal escape sequence
#endif

extern void initialize_images_treectrl(void);

extern wxBitmap img_lightning_s;
extern wxBitmap img_other_s;
extern wxBitmap img_precipitation_s;
extern wxBitmap img_temperature_s;
extern wxBitmap img_wind_s;

#endif /* _TREECTRL_H_ */
