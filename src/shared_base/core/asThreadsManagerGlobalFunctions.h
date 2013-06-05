/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASTHREADSMANAGERGLOBALFUNCTIONS_H
#define ASTHREADSMANAGERGLOBALFUNCTIONS_H

#include <asIncludes.h>

class asThreadsManager;

// Threads manager
extern asThreadsManager* g_pThreadsManager;
asThreadsManager& ThreadsManager();
void DeleteThreadsManager();

#endif // ASTHREADSMANAGERGLOBALFUNCTIONS_H
